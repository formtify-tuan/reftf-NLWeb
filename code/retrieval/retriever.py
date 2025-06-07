"""
Simplified vector database interface using Qdrant.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Union

from config.config import CONFIG
from utils.utils import get_param
from utils.logging_config_helper import get_configured_logger
from utils.logger import LogLevel

from retrieval.qdrant import QdrantVectorClient

logger = get_configured_logger("retriever")


class VectorDBClient:
    """Client wrapper around :class:`QdrantVectorClient`."""

    def __init__(self, endpoint_name: Optional[str] = None, query_params: Optional[Dict[str, Any]] = None):
        self.query_params = query_params or {}
        self.endpoint_name = endpoint_name or CONFIG.preferred_retrieval_endpoint

        if CONFIG.is_development_mode() and self.query_params:
            self.endpoint_name = get_param(self.query_params, "db", str, self.endpoint_name)
            logger.debug(f"Development mode: endpoint overridden to {self.endpoint_name}")

        if self.endpoint_name not in CONFIG.retrieval_endpoints:
            error_msg = f"Invalid endpoint: {self.endpoint_name}. Must be one of: {list(CONFIG.retrieval_endpoints.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.client = QdrantVectorClient(self.endpoint_name)
        self.db_type = "qdrant"
        self._retrieval_lock = asyncio.Lock()

        logger.info(f"VectorDBClient initialized - endpoint: {self.endpoint_name}")

    async def delete_documents_by_site(self, site: str, **kwargs) -> int:
        async with self._retrieval_lock:
            logger.info(f"Deleting documents for site: {site}")
            try:
                count = await self.client.delete_documents_by_site(site, **kwargs)
                logger.info(f"Successfully deleted {count} documents for site: {site}")
                return count
            except Exception as e:
                logger.exception(f"Error deleting documents for site {site}: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Document deletion failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "site": site,
                        "endpoint": self.endpoint_name,
                    },
                )
                raise

    async def upload_documents(self, documents: List[Dict[str, Any]], **kwargs) -> int:
        async with self._retrieval_lock:
            logger.info(f"Uploading {len(documents)} documents")
            try:
                count = await self.client.upload_documents(documents, **kwargs)
                logger.info(f"Successfully uploaded {count} documents")
                return count
            except Exception as e:
                logger.exception(f"Error uploading documents: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Document upload failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "document_count": len(documents),
                        "endpoint": self.endpoint_name,
                    },
                )
                raise

    async def search(
        self,
        query: str,
        site: Union[str, List[str]],
        num_results: int = 50,
        **kwargs,
    ) -> List[List[str]]:
        if site == "all":
            sites = CONFIG.nlweb.sites
            if len(sites) == 0 or sites == "all":
                return await self.search_all_sites(query, num_results, **kwargs)
            site = sites

        if isinstance(site, str) and "," in site:
            site = site.replace("[", "").replace("]", "")
            site = [s.strip() for s in site.split(",")]
        elif isinstance(site, str):
            site = site.replace(" ", "_")

        async with self._retrieval_lock:
            logger.info(f"Searching for '{query[:50]}...' in site: {site}, num_results: {num_results}")
            start_time = time.time()
            try:
                results = await self.client.search(query, site, num_results, **kwargs)
                search_duration = time.time() - start_time
                logger.log_with_context(
                    LogLevel.INFO,
                    "Search completed",
                    {
                        "duration": f"{search_duration:.2f}s",
                        "results_count": len(results),
                        "site": site,
                    },
                )
                return results
            except Exception as e:
                logger.exception(f"Error in search: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Search failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "site": site,
                        "endpoint": self.endpoint_name,
                    },
                )
                raise

    async def search_by_url(self, url: str, **kwargs) -> Optional[List[str]]:
        async with self._retrieval_lock:
            logger.info(f"Retrieving item with URL: {url}")
            try:
                result = await self.client.search_by_url(url, **kwargs)
                if result:
                    logger.debug(f"Successfully retrieved item for URL: {url}")
                else:
                    logger.warning(f"No item found for URL: {url}")
                return result
            except Exception as e:
                logger.exception(f"Error retrieving item with URL: {url}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Item retrieval failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "url": url,
                        "endpoint": self.endpoint_name,
                    },
                )
                raise

    async def search_all_sites(self, query: str, num_results: int = 50, **kwargs) -> List[List[str]]:
        async with self._retrieval_lock:
            logger.info(f"Searching across all sites for '{query[:50]}...', num_results: {num_results}")
            start_time = time.time()
            try:
                results = await self.client.search_all_sites(query, num_results, **kwargs)
                search_duration = time.time() - start_time
                logger.log_with_context(
                    LogLevel.INFO,
                    "All-sites search completed",
                    {
                        "duration": f"{search_duration:.2f}s",
                        "results_count": len(results),
                    },
                )
                return results
            except Exception as e:
                logger.exception(f"Error in search_all_sites: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "All-sites search failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "endpoint": self.endpoint_name,
                    },
                )
                raise


def get_vector_db_client(endpoint_name: Optional[str] = None, query_params: Optional[Dict[str, Any]] = None) -> VectorDBClient:
    """Factory helper to create :class:`VectorDBClient`."""
    return VectorDBClient(endpoint_name=endpoint_name, query_params=query_params)
