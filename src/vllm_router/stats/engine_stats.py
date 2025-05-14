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
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx
from prometheus_client.parser import text_string_to_metric_families

from vllm_router.httpx_client import HTTPXClientWrapper
from vllm_router.log import init_logger
from vllm_router.service_discovery import get_service_discovery
from vllm_router.utils import SingletonMeta

logger = init_logger(__name__)


@dataclass
class EngineStats:
    # Number of running requests
    num_running_requests: int = 0
    # Number of queuing requests
    num_queuing_requests: int = 0
    # GPU prefix cache hit rate (as used in some panels)
    gpu_prefix_cache_hit_rate: float = 0.0
    # GPU KV usage percentage (new field for dashboard "GPU KV Usage Percentage")
    gpu_cache_usage_perc: float = 0.0

    @staticmethod
    def from_vllm_scrape(vllm_scrape: str):
        """
        Parse the vllm scrape string and return a EngineStats object

        Args:
            vllm_scrape (str): The vllm scrape string

        Returns:
            EngineStats: The EngineStats object

        Note:
            Assume vllm only runs a single model
        """
        num_running_reqs = 0
        num_queuing_reqs = 0
        gpu_prefix_cache_hit_rate = 0.0
        gpu_cache_usage_perc = 0.0

        for family in text_string_to_metric_families(vllm_scrape):
            for sample in family.samples:
                if sample.name == "vllm:num_requests_running":
                    num_running_reqs = sample.value
                elif sample.name == "vllm:num_requests_waiting":
                    num_queuing_reqs = sample.value
                elif sample.name == "vllm:gpu_prefix_cache_hit_rate":
                    gpu_prefix_cache_hit_rate = sample.value
                elif sample.name == "vllm:gpu_cache_usage_perc":
                    gpu_cache_usage_perc = sample.value

        return EngineStats(
            num_running_requests=num_running_reqs,
            num_queuing_requests=num_queuing_reqs,
            gpu_prefix_cache_hit_rate=gpu_prefix_cache_hit_rate,
            gpu_cache_usage_perc=gpu_cache_usage_perc,
        )


class EngineStatsScraper(metaclass=SingletonMeta):
    def __init__(
        self,
        scrape_interval: float = None,
        httpx_client_wrapper: Optional[HTTPXClientWrapper] = None,
    ):
        """
        Initialize the scraper to periodically fetch metrics from all serving engines.

        Args:
            scrape_interval (float): The interval in seconds
                to scrape the metrics.
            httpx_client_wrapper (Optional[HTTPXClientWrapper]): The HTTPX client wrapper
                for making requests.

        Raises:
            ValueError: if the service discover module is have
            not been initialized.

        """
        if hasattr(self, "_initialized"):
            # If already initialized, ensure scrape_interval and httpx_client_wrapper are not unexpectedly changed.
            # Or, decide if they can be reconfigured (current design implies not easily).
            if scrape_interval is not None and self.scrape_interval != scrape_interval:
                logger.warning(
                    f"EngineStatsScraper already initialized. Ignoring new scrape_interval: {scrape_interval}"
                )
            # httpx_client_wrapper is harder to check for change if it's an object, assume it's set on first init.
            return

        if scrape_interval is None:
            raise ValueError(
                "EngineStatsScraper must be initialized with scrape_interval on its first call."
            )

        self.engine_stats: Dict[str, EngineStats] = {}
        self.engine_stats_lock = threading.Lock()
        self.scrape_interval = scrape_interval

        if httpx_client_wrapper:
            self.httpx_client = httpx_client_wrapper()  # Use __call__ method directly
        else:
            logger.warning("EngineStatsScraper: No wrapper, creating own httpx client.")
            self.httpx_client = httpx.AsyncClient()
        self._created_own_httpx_client = not bool(httpx_client_wrapper)

        # scrape thread
        self.running = True
        self.scrape_thread = threading.Thread(target=self._scrape_worker, daemon=True)
        self.scrape_thread.start()
        self._initialized = True

    async def _scrape_one_endpoint(self, url: str) -> Optional[EngineStats]:
        """
        Scrape metrics from a single serving engine.

        Args:
            url (str): The URL of the serving engine (does not contain endpoint)
        """
        try:
            # Use a timeout slightly less than scrape_interval to avoid overlapping scrapes
            # or ensure timely completion within the interval.
            timeout_duration = (
                self.scrape_interval * 0.9
                if self.scrape_interval > 1
                else self.scrape_interval
            )
            timeout_config = httpx.Timeout(
                timeout_duration, connect=min(timeout_duration, 3.0)
            )

            response = await self.httpx_client.get(
                url + "/metrics", timeout=timeout_config
            )
            response.raise_for_status()
            engine_stats = EngineStats.from_vllm_scrape(response.text)
        except httpx.TimeoutException as e:
            logger.error(f"HTTPX Timeout scraping metrics from {url}: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"HTTPX RequestError scraping metrics from {url}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error(f"HTTPX StatusErr {status} scraping {url}: {e.response.text}")
            return None
        except ValueError as e:  # For EngineStats.from_vllm_scrape parsing issues
            logger.error(f"ValueError parsing metrics from {url}: {e}")
            return None
        except Exception as e:
            err_type = type(e).__name__
            logger.error(
                f"Unexpected err scraping metrics from {url}: {err_type} - {e}"
            )
            return None
        return engine_stats

    async def _scrape_endpoints(self):
        """
        Scrape metrics from all serving engines. Get a list of backend endpoint
        from the service discovery module, then scrape the metrics
        endpoint on each of them. The metrics are
        stored in self.engine_stats.

        """
        collected_engine_stats = {}
        endpoints = get_service_discovery().get_endpoint_info()
        logger.info(f"Scraping metrics from {len(endpoints)} serving engine(s)")

        tasks = []
        for info in endpoints:
            url = info.url
            tasks.append(self._scrape_one_endpoint(url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_collected_engine_stats = {}
        for i, result in enumerate(results):
            url = endpoints[i].url  # Get corresponding url
            if isinstance(result, EngineStats):
                new_collected_engine_stats[url] = result
            elif isinstance(result, Exception):
                # Errors are already logged in _scrape_one_endpoint
                pass  # Or log a summary error here if needed

        with self.engine_stats_lock:
            old_urls = list(self.engine_stats.keys())
            for old_url in old_urls:
                if old_url not in new_collected_engine_stats:
                    del self.engine_stats[old_url]
            for url, stats in new_collected_engine_stats.items():
                self.engine_stats[url] = stats

    def _sleep_or_break(self, check_interval: float = 1):
        """
        Sleep for self.scrape_interval seconds if self.running is True.
        Otherwise, break the loop.
        """
        for _ in range(int(self.scrape_interval / check_interval)):
            if not self.running:
                break
            time.sleep(check_interval)

    async def _scrape_worker(self):
        """
        Periodically scrape metrics from all serving engines, and update
        engine stats. It scrapes metrics with regular intervals as defined
        in self.scrape_interval.
        """
        while self.running:
            try:
                await self._scrape_endpoints()
            except Exception as e:
                logger.error(f"Failed to scrape endpoints: {e}")
                # If there was an error, sleep and try again. This can cover
                # transient errors, but not systematic errors like incorrect configuration.
                # The latter would require more fallback logic or eventual system shutdown.
            finally:
                # Whether we succeeded or failed, we sleep for the scrape interval
                self._sleep_or_break()

    def _async_scrape_worker(self):
        """
        A wrapper function for the async _scrape_worker function. This is
        what gets run in the background thread. It runs _scrape_worker
        continuously untial self.running is False.
        """
        asyncio.run(self._scrape_worker())

    def get_engine_stats(self) -> Dict[str, EngineStats]:
        """
        Get the engine stats.

        Returns:
            Dict[str, EngineStats]: Return a copy of the engine stats
        """
        with self.engine_stats_lock:
            return dict(self.engine_stats)

    def get_health(self) -> bool:
        """
        Check if the stats scraper is healthy.

        Returns:
            bool: True if the stats scraper is healthy, False otherwise
        """
        return self.running and self.scrape_thread.is_alive()

    def close(self):
        """
        Stop the background thread and cleanup resources.
        """
        self.running = False
        self.scrape_thread.join()
        if self._created_own_httpx_client and self.httpx_client:
            try:
                asyncio.run(self.httpx_client.aclose())
                logger.info("EngineStatsScraper closed its own httpx client.")
            except RuntimeError as e:
                logger.error(f"RuntimeErr closing own httpx client: {e}")
            except Exception as e:
                err_type = type(e).__name__
                logger.error(f"Err closing own httpx client: {err_type} - {e}")


def initialize_engine_stats_scraper(
    scrape_interval: float, httpx_client_wrapper: Optional[HTTPXClientWrapper] = None
) -> EngineStatsScraper:
    return EngineStatsScraper(scrape_interval, httpx_client_wrapper)


def get_engine_stats_scraper() -> EngineStatsScraper:
    # This call returns the already-initialized instance or raises if scrape_interval was not provided on first call.
    # It relies on SingletonMeta to not re-initialize if already done.
    # If httpx_client_wrapper needs to be passed here too, it becomes more complex as get_... usually doesn't take params.
    # The current design requires httpx_client_wrapper to be passed during initialize_engine_stats_scraper.
    return EngineStatsScraper()
