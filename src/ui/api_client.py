# src/ui/api_client.py
"""
Synchronous HTTP client for communicating with the FastAPI backend.
Provides methods for ingestion and querying.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests

logger = logging.getLogger("ui.api_client")


class APIClient:
    """Client for interacting with the RAG API."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8000/api/v1)
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # Set default timeout to 120 seconds for all requests
        self.session.request = lambda method, url, **kwargs: requests.request(
            method, url, timeout=120, **kwargs
        )

    def ingest_files(self, file_paths: List[Path]) -> dict:
        """
        Upload files for ingestion.

        Args:
            file_paths: List of paths to files to upload

        Returns:
            JSON response from the API
        """
        try:
            files = []
            for file_path in file_paths:
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                # Open file in binary mode
                files.append(
                    ("files", (file_path.name, open(file_path, "rb"), "application/octet-stream"))
                )

            response = self.session.post(
                f"{self.base_url}/ingest",
                files=files
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            logger.error(f"HTTP error during ingestion: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API error: {e.response.status_code} - {e.response.text}")
        except requests.RequestException as e:
            logger.error(f"Request error during ingestion: {e}")
            raise Exception(f"Failed to connect to API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}")
            raise
        finally:
            # Close opened files
            for _, (_, f, _) in files:
                f.close()

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Send a query to the API.

        Args:
            question: The user's question
            top_k: Number of results to retrieve

        Returns:
            JSON response from the API
        """
        try:
            payload = {
                "query": question,
                "top_k": top_k
            }
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            logger.error(f"HTTP error during query: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API error: {e.response.status_code} - {e.response.text}")
        except requests.RequestException as e:
            logger.error(f"Request error during query: {e}")
            raise Exception(f"Failed to connect to API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise

    def health_check(self) -> dict:
        """
        Check the health of the API.

        Returns:
            JSON response from the health endpoint
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            logger.error(f"HTTP error during health check: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API health check failed: {e.response.status_code} - {e.response.text}")
        except requests.RequestException as e:
            logger.error(f"Request error during health check: {e}")
            raise Exception(f"Failed to connect to API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            raise

    def close(self):
        """Close the HTTP client."""
        self.session.close()


# Singleton instance for use in Streamlit
_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """Get or create the singleton API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client