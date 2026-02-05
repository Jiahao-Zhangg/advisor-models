"""Client for remote SWE-Smith evaluation.

This client sends evaluation requests to a remote server running eval_server.py.
Used when Docker is not available locally (e.g., on Mosaic).
"""

import os
import logging
import requests
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class RemoteEvaluationClient:
    """Client for communicating with remote evaluation server."""

    def __init__(self, server_url: str = None, timeout: int = 300):
        """Initialize the client.

        Args:
            server_url: URL of the evaluation server (e.g., http://YOUR_VM_IP:8082)
                       If None, reads from EVAL_SERVER_URL environment variable
            timeout: Maximum time to wait for evaluation (seconds)
        """
        self.server_url = server_url or os.environ.get("EVAL_SERVER_URL")
        if not self.server_url:
            raise ValueError(
                "server_url must be provided or EVAL_SERVER_URL environment variable must be set"
            )

        # Remove trailing slash
        self.server_url = self.server_url.rstrip("/")
        self.timeout = timeout

        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to evaluation server at {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to evaluation server: {e}")
            raise

    def _serialize_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Convert instance to JSON-serializable format.

        Handles numpy arrays and other non-serializable types.
        """
        import numpy as np

        serialized = {}
        for key, value in instance.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_instance(value)
            elif isinstance(value, list):
                serialized[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    def compute_score(
        self, patch: str, instance: Dict[str, Any]
    ) -> Tuple[float, str, str]:
        """Compute score by sending to remote evaluation server.

        Args:
            patch: The generated patch string
            instance: The problem instance dict from SWE-Smith dataset

        Returns:
            Tuple of (reward, run_id, info_string)
        """
        try:
            # Serialize instance to handle numpy arrays
            serialized_instance = self._serialize_instance(instance)

            # Use synchronous endpoint for simplicity
            response = requests.post(
                f"{self.server_url}/evaluate_sync",
                json={"patch": patch, "instance": serialized_instance},
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            if "error" in result:
                return 0.0, "", f"Remote evaluation failed: {result['error']}"

            return result["reward"], result["run_id"], result["info"]

        except requests.exceptions.Timeout:
            logger.error(
                f"Evaluation timeout for instance {instance.get('instance_id', 'unknown')}"
            )
            return 0.0, "ERROR", f"Evaluation timeout after {self.timeout}s"
        except Exception as e:
            logger.error(f"Remote evaluation error: {e}")
            return 0.0, "ERROR", f"Remote evaluation error: {str(e)}"


def compute_score_remote(
    patch: str, instance: Dict[str, Any]
) -> Tuple[float, str, str]:
    """Convenience function that creates client and computes score.

    Uses EVAL_SERVER_URL environment variable for server location.
    """
    client = RemoteEvaluationClient()
    return client.compute_score(patch, instance)
