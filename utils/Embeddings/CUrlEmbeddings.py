import logging
import json
import subprocess
from typing import List
from langchain_core.embeddings import Embeddings


class CurlEmbeddings(Embeddings):
    """Custom embeddings implementation using curl for HTTP requests."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "text-embedding-ada-002",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = kwargs.get("timeout", 30)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using curl."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Embedding {len(texts)} documents using curl")

        embeddings = []

        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using curl."""
        logger = logging.getLogger(__name__)
        logger.debug("Embedding query using curl")

        embeddings = self._embed_batch([text])
        return embeddings[0] if embeddings else []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using curl."""
        logger = logging.getLogger(__name__)

        try:
            # Prepare the request payload
            payload = {"model": self.model_name, "input": texts}

            # Prepare curl command
            curl_cmd = [
                "curl",
                "-s",  # Silent mode
                "-X",
                "POST",
                "--ssl-no-revoke",  # Windows SSL fix
                "--tlsv1.2",  # Use TLS 1.2
                "-H",
                "Content-Type: application/json",
                "-H",
                f"Authorization: Bearer {self.api_key}",
                "--connect-timeout",
                str(self.timeout),
                "--max-time",
                str(self.timeout * 2),
                "-d",
                json.dumps(payload),
                self.api_url,
            ]

            # Execute curl command
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace invalid characters instead of failing
                timeout=self.timeout * 2,
            )

            if result.returncode != 0:
                logger.error(
                    f"Curl embeddings command failed with return code {result.returncode}"
                )
                logger.error(f"Stderr: {result.stderr}")
                raise Exception(f"Curl embeddings request failed: {result.stderr}")

            # Parse the response
            response_data = json.loads(result.stdout)

            if "error" in response_data:
                raise Exception(f"Embeddings API Error: {response_data['error']}")

            if "data" in response_data:
                embeddings = [item["embedding"] for item in response_data["data"]]
                logger.debug(f"Generated {len(embeddings)} embeddings for batch")
                return embeddings
            else:
                raise Exception("No valid embeddings data in response")

        except subprocess.TimeoutExpired:
            logger.error("Curl embeddings request timed out")
            raise Exception("Embeddings request timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse embeddings JSON response: {e}")
            logger.error(f"Raw response: {result.stdout}")
            raise Exception(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Curl embeddings request failed: {e}")
            raise
