# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import asyncio
import enum
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from kubernetes import client, config, watch

from vllm_router.httpx_client import HTTPXClientWrapper
from vllm_router.log import init_logger

logger = init_logger(__name__)

_global_service_discovery: "Optional[ServiceDiscovery]" = None


class ServiceDiscoveryType(enum.Enum):
    STATIC = "static"
    K8S = "k8s"


@dataclass
class EndpointInfo:
    url: str
    model_name: Optional[str] = None
    added_timestamp: Optional[int] = None
    load: Optional[float] = None


class ServiceDiscovery(abc.ABC):
    @abc.abstractmethod
    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get a list of serving endpoints.

        Returns:
            a list of serving endpoints
        """
        pass

    def close(self) -> None:
        """
        Close the service discovery.
        """
        pass

    @abc.abstractmethod
    def get_health(self) -> bool:
        """
        Check if the service discovery is healthy.

        Returns:
            True if the service discovery is healthy, False otherwise
        """
        pass


class StaticServiceDiscovery(ServiceDiscovery):
    def __init__(self, engine_urls: List[str], model_names: List[str] = None):
        """
        Initialize the static service discovery module.

        Args:
            engine_urls: a list of engine URLs
            model_names: a list of model names, default to None
                (unknown)
        """
        if len(engine_urls) == 0:
            raise ValueError(
                "No engine URLs provided. Please provide at least one engine URL."
            )

        if model_names is None:
            model_names = [None] * len(engine_urls)
        elif len(model_names) != len(engine_urls):
            raise ValueError(
                f"The number of model names ({len(model_names)}) "
                f"must be equal to the number of engine URLs ({len(engine_urls)})."
            )
        current_ts = int(time.time())
        self.available_engines = [
            EndpointInfo(url=url, model_name=name, added_timestamp=current_ts)
            for url, name in zip(engine_urls, model_names)
        ]

    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get a list of serving endpoints.

        Returns:
            a list of serving endpoints
        """
        return self.available_engines

    def get_health(self) -> bool:
        """
        Check if the service discovery is healthy.

        Returns:
            True if the service discovery is healthy, False otherwise
        """
        return True


class K8sServiceDiscovery(ServiceDiscovery):
    def __init__(
        self,
        namespace: str,
        port: str,
        label_selector=None,
        httpx_client_wrapper: Optional[HTTPXClientWrapper] = None,
    ):
        """
        Initialize the Kubernetes service discovery module. This module
        assumes all serving engine pods are in the same namespace, listening
        on the same port, and have the same label selector.

        It will start a daemon thread to watch the engine pods and update
        the url of the available engines.

        Args:
            namespace: the namespace of the engine pods
            port: the port of the engines
            label_selector: the label selector of the engines
            httpx_client_wrapper: An optional HTTPXClientWrapper instance.
        """
        self.namespace = namespace
        self.port = port
        self.available_engines: Dict[str, EndpointInfo] = {}
        self.available_engines_lock = threading.Lock()
        self.label_selector = label_selector

        if httpx_client_wrapper:
            self.httpx_client = httpx_client_wrapper()  # Use __call__ method directly
        else:
            logger.warning(
                "K8sServiceDiscovery: No wrapper, creating own httpx client."
            )
            self.httpx_client = httpx.AsyncClient()
        self._created_own_httpx_client = not bool(httpx_client_wrapper)

        # Init kubernetes watcher
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.k8s_api = client.CoreV1Api()
        self.k8s_watcher = watch.Watch()

        # Start watching engines
        self.running = True
        self.watcher_thread = threading.Thread(target=self._watch_engines, daemon=True)
        self.watcher_thread.start()

    @staticmethod
    def _check_pod_ready(container_statuses):
        """
        Check if all containers in the pod are ready.

        Args:
            container_statuses: a list of container statuses from
                k8s container statuses.
        """
        if not container_statuses:
            return False
        ready_count = sum(1 for status in container_statuses if status.ready)
        return ready_count == len(container_statuses)

    async def _get_model_name(self, pod_ip: str) -> Optional[str]:
        """
        Get the model name of the serving engine pod by querying the pod's
        '/v1/models' endpoint asynchronously.

        Args:
            pod_ip: the IP address of the pod

        Returns:
            the model name of the serving engine
        """
        url = f"http://{pod_ip}:{self.port}/v1/models"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

            # Using a short timeout to prevent long hangs.
            # Consider making these timeouts configurable.
            timeout_config = httpx.Timeout(10.0, connect=3.0)  # 10s total, 3s connect

            response = await self.httpx_client.get(
                url, headers=headers, timeout=timeout_config
            )
            response.raise_for_status()
            model_name_data = response.json()
            if (
                model_name_data
                and "data" in model_name_data
                and isinstance(model_name_data["data"], list)
                and len(model_name_data["data"]) > 0
            ):
                if (
                    isinstance(model_name_data["data"][0], dict)
                    and "id" in model_name_data["data"][0]
                ):
                    return model_name_data["data"][0]["id"]
        except httpx.TimeoutException as e:
            logger.error(f"HTTPX Timeout querying model info from {url}: {e}")
        except httpx.RequestError as e:
            logger.error(f"HTTPX RequestError querying model info from {url}: {e}")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error(
                f"HTTPX StatusErr {status} querying model info from {url}: {e.response.text}"
            )
        except ValueError as e:  # JSON parsing errors
            logger.error(f"ValueError parsing model info response from {url}: {e}")
        except Exception as e:
            err_type = type(e).__name__
            logger.error(
                f"Unexpected err querying model info from {url}: {err_type} - {e}"
            )
        return None

    def _watch_engines(self):
        """
        Watch the engine pods and update the available engines.
        """
        while self.running:
            try:
                for event in self.k8s_watcher.stream(
                    self.k8s_api.list_namespaced_pod,
                    namespace=self.namespace,
                    label_selector=self.label_selector,
                ):
                    pod = event["object"]
                    event_type = event["type"]
                    pod_name = pod.metadata.name
                    pod_ip = pod.status.pod_ip
                    is_pod_ready = self._check_pod_ready(pod.status.container_statuses)
                    model_name = None
                    if is_pod_ready and pod_ip:
                        try:
                            model_name = asyncio.run(self._get_model_name(pod_ip))
                        except RuntimeError as e:
                            logger.error(
                                f"RuntimeErr for _get_model_name({pod_ip}): {e}. Check async setup."
                            )
                            model_name = None
                        except Exception as e:  # General catch-all
                            err_type = type(e).__name__
                            logger.error(
                                f"Err calling _get_model_name({pod_ip}): {err_type} - {e}"
                            )
                            model_name = None
                    else:
                        model_name = None
                    self._on_engine_update(
                        pod_name, pod_ip, event_type, is_pod_ready, model_name
                    )
            except Exception as e:
                logger.error(f"K8s watcher error: {e}")
                time.sleep(0.5)

    def _add_engine(self, engine_name: str, engine_ip: str, model_name: str):
        logger.info(
            f"Discovered new serving engine {engine_name} at {engine_ip}, running model: {model_name}"
        )
        with self.available_engines_lock:
            self.available_engines[engine_name] = EndpointInfo(
                url=f"http://{engine_ip}:{self.port}",
                model_name=model_name,
                added_timestamp=int(time.time()),
            )

    def _delete_engine(self, engine_name: str):
        logger.info(f"Serving engine {engine_name} is deleted")
        with self.available_engines_lock:
            del self.available_engines[engine_name]

    def _on_engine_update(
        self,
        engine_name: str,
        engine_ip: Optional[str],
        event: str,
        is_pod_ready: bool,
        model_name: Optional[str],
    ) -> None:
        if event == "ADDED":
            if engine_ip is None:
                return

            if not is_pod_ready:
                return

            if model_name is None:
                return

            self._add_engine(engine_name, engine_ip, model_name)

        elif event == "DELETED":
            if engine_name not in self.available_engines:
                return

            self._delete_engine(engine_name)

        elif event == "MODIFIED":
            if engine_ip is None:
                return

            if is_pod_ready and model_name is not None:
                self._add_engine(engine_name, engine_ip, model_name)
                return

            if (
                not is_pod_ready or model_name is None
            ) and engine_name in self.available_engines:
                self._delete_engine(engine_name)
                return

    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Returns:
            a list of engine URLs
        """
        with self.available_engines_lock:
            return list(self.available_engines.values())

    def get_health(self) -> bool:
        """
        Check if the service discovery module is healthy.

        Returns:
            True if the service discovery module is healthy, False otherwise
        """
        return self.watcher_thread.is_alive()

    def close(self) -> None:
        """
        Close the service discovery.
        """
        self.running = False
        self.watcher_thread.join()
        if self._created_own_httpx_client and hasattr(self.httpx_client, "aclose"):
            try:
                asyncio.run(self.httpx_client.aclose())
                logger.info("K8sServiceDiscovery closed its own httpx client")
            except RuntimeError as e:
                logger.error(f"RuntimeErr closing own httpx client: {e}")
            except Exception as e:
                err_type = type(e).__name__
                logger.error(f"Err closing own httpx client: {err_type} - {e}")
        self.k8s_watcher.stop()


def initialize_service_discovery(
    discovery_type: ServiceDiscoveryType,
    static_backends: Optional[List[str]] = None,
    static_models: Optional[List[str]] = None,
    k8s_namespace: Optional[str] = None,
    k8s_port: Optional[str] = None,
    k8s_label_selector: Optional[str] = None,
    httpx_client_wrapper: Optional[HTTPXClientWrapper] = None,
) -> ServiceDiscovery:
    """
    Initialize the service discovery module.

    Args:
        discovery_type: the type of service discovery
        static_backends: a list of static backends, required if discovery_type is STATIC
        static_models: a list of static models, optional if discovery_type is STATIC
        k8s_namespace: the namespace of vLLM pods, required if discovery_type is K8S
        k8s_port: the port of vLLM processes, required if discovery_type is K8S
        k8s_label_selector: the label selector for vLLM pods, optional if discovery_type is K8S
        httpx_client_wrapper: An optional HTTPXClientWrapper instance.

    Returns:
        the initialized service discovery object

    Raises:
        ValueError: if required arguments are missing
    """
    global _global_service_discovery
    if _global_service_discovery is not None:
        return _global_service_discovery

    if discovery_type == ServiceDiscoveryType.STATIC:
        if static_backends is None:
            raise ValueError(
                "static_backends is required for STATIC service discovery."
            )
        _global_service_discovery = StaticServiceDiscovery(
            static_backends, static_models
        )
    elif discovery_type == ServiceDiscoveryType.K8S:
        if k8s_namespace is None:
            raise ValueError("k8s_namespace is required for K8S service discovery.")
        if k8s_port is None:
            raise ValueError("k8s_port is required for K8S service discovery.")
        _global_service_discovery = K8sServiceDiscovery(
            k8s_namespace, k8s_port, k8s_label_selector, httpx_client_wrapper
        )
    else:
        raise ValueError(f"Unknown service discovery type: {discovery_type}")

    return _global_service_discovery


def reconfigure_service_discovery(
    discovery_type: ServiceDiscoveryType,
    static_backends: Optional[List[str]] = None,
    static_models: Optional[List[str]] = None,
    k8s_namespace: Optional[str] = None,
    k8s_port: Optional[str] = None,
    k8s_label_selector: Optional[str] = None,
    httpx_client_wrapper: Optional[HTTPXClientWrapper] = None,
) -> ServiceDiscovery:
    """
    Reconfigure the service discovery module with the given parameters.

    This stops the current service discovery and creates a new one with the provided parameters.

    Args:
        discovery_type: the type of service discovery
        static_backends: a list of static backends, required if discovery_type is STATIC
        static_models: a list of static models, optional if discovery_type is STATIC
        k8s_namespace: the namespace of vLLM pods, required if discovery_type is K8S
        k8s_port: the port of vLLM processes, required if discovery_type is K8S
        k8s_label_selector: the label selector for vLLM pods, optional if discovery_type is K8S
        httpx_client_wrapper: An optional HTTPXClientWrapper instance.

    Returns:
        the reconfigured service discovery object

    Raises:
        ValueError: if the service discovery object has not been initialized
    """
    global _global_service_discovery
    if _global_service_discovery is None:
        raise ValueError("Service discovery has not been initialized.")

    # Close the current service discovery
    _global_service_discovery.close()
    _global_service_discovery = None

    # Initialize a new service discovery
    return initialize_service_discovery(
        discovery_type,
        static_backends,
        static_models,
        k8s_namespace,
        k8s_port,
        k8s_label_selector,
        httpx_client_wrapper,
    )


def get_service_discovery() -> ServiceDiscovery:
    """
    Get the service discovery object.

    Returns:
        the service discovery object

    Raises:
        ValueError: if the service discovery object has not been initialized
    """
    global _global_service_discovery
    if _global_service_discovery is None:
        raise ValueError(
            "Service discovery has not been initialized. Call initialize_service_discovery first."
        )
    return _global_service_discovery


if __name__ == "__main__":
    # Test the service discovery
    # k8s_sd = K8sServiceDiscovery("default", 8000, "release=test")
    initialize_service_discovery(
        ServiceDiscoveryType.K8S,
        namespace="default",
        port=8000,
        label_selector="release=test",
    )

    k8s_sd = get_service_discovery()

    time.sleep(1)
    while True:
        urls = k8s_sd.get_endpoint_info()
        print(urls)
        time.sleep(2)
