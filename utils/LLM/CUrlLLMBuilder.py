import logging
import os
import subprocess
import json
from typing import Any, List, Optional, Iterator
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk, LLMResult, Generation
from pydantic import Field

logger = logging.getLogger(__name__)


class CurlLLM(LLM):
    """
    A custom LLM implementation using curl for HTTP requests.
    This can be used to interface with various API endpoints.
    """

    api_url: str = Field(..., description="The API endpoint URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    model_name: str = Field("curl-model", description="Model name identifier")
    temperature: float = Field(0.7, description="Temperature for response generation")
    max_tokens: int = Field(2000, description="Maximum tokens in response (increased for DeepSeek R1 reasoning + content)")
    timeout: int = Field(30, description="Request timeout in seconds")
    headers: dict = Field(default_factory=dict, description="Additional HTTP headers")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return identifier of LLM type."""
        return "curl_llm"

    def _prepare_headers(self) -> dict:
        """Prepare HTTP headers for the request."""
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Add API key if provided
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"

        # Merge with custom headers
        default_headers.update(self.headers)
        return default_headers

    def _prepare_payload(self, prompt: str) -> dict:
        """
        Prepare the request payload.
        Override this method for different API formats.
        """
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,  # Ensure we get the complete response
        }

    def _make_curl_request(self, payload: dict) -> str:
        """Make a curl request to the API endpoint."""
        headers = self._prepare_headers()

        # Build curl command
        curl_cmd = ["curl", "-s", "-X", "POST"]

        # Add SSL options for Windows compatibility
        curl_cmd.extend(["--ssl-no-revoke", "--tlsv1.2"])

        # Add timeout
        curl_cmd.extend(["--max-time", str(self.timeout)])

        # Add headers with proper quoting for Windows
        for key, value in headers.items():
            curl_cmd.extend(["-H", f"{key}: {value}"])

        # Add data
        curl_cmd.extend(["-d", json.dumps(payload)])

        # Add URL
        curl_cmd.append(self.api_url)

        try:
            logger.debug(
                f"Executing curl command: {' '.join(curl_cmd[:10])}..."
            )  # Log partial command for security

            # Execute curl command
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of failing
                timeout=self.timeout + 5,  # Add buffer to subprocess timeout
            )

            if result.returncode != 0:
                error_msg = f"Curl command failed with return code {result.returncode}: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Check if stdout is None (can happen with encoding issues)
            if result.stdout is None:
                error_msg = "Curl command returned None output (possible encoding issue)"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.debug(
                f"Curl request successful, response length: {len(result.stdout)}"
            )
            return result.stdout

        except subprocess.TimeoutExpired:
            error_msg = f"Curl request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Curl request failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _parse_response(self, response_text: str) -> str:
        """
        Parse the API response to extract the generated text.
        Override this method for different API response formats.
        """
        try:
            # Clean up response text - remove leading/trailing whitespace and newlines
            cleaned_response = response_text.strip()

            # Try to find JSON in the response (sometimes there might be extra whitespace)
            start_idx = cleaned_response.find("{")
            if start_idx != -1:
                cleaned_response = cleaned_response[start_idx:]

            response_data = json.loads(cleaned_response)
            logger.debug(
                f"Parsed response data: {json.dumps(response_data, indent=2)[:200]}..."
            )

            # Common response formats - adapt as needed
            if "choices" in response_data and response_data["choices"]:
                # OpenAI-style response (check for both text and message content)
                choice = response_data["choices"][0]
                if "message" in choice:
                    message = choice["message"]
                    # For DeepSeek R1: Always prioritize content over reasoning
                    # The content field contains the final answer after reasoning is complete
                    if "content" in message and message["content"].strip():
                        content = message["content"].strip()
                        logger.debug(
                            f"Extracted content from message: {content[:100]}..."
                        )
                        # Also log that reasoning was available if present
                        if "reasoning" in message and message["reasoning"]:
                            logger.debug(f"Reasoning was present (length: {len(message['reasoning'])} chars)")
                        return content
                    # Only use reasoning as fallback if content is truly empty
                    elif "reasoning" in message and message["reasoning"]:
                        reasoning = message["reasoning"].strip()
                        logger.warning(
                            "Content field is empty, using reasoning as fallback (this may indicate incomplete response)"
                        )
                        logger.debug(
                            f"Extracted reasoning from message: {reasoning[:100]}..."
                        )
                        return reasoning
                elif "text" in choice:
                    text = choice["text"].strip()
                    logger.debug(f"Extracted text from choice: {text[:100]}...")
                    return text
                else:
                    logger.warning(f"Unexpected choice format: {choice}")
                    return str(choice).strip()
            elif "response" in response_data:
                # Simple response format
                return response_data["response"].strip()
            elif "content" in response_data:
                # Content-based response
                return response_data["content"].strip()
            elif "text" in response_data:
                # Direct text response
                return response_data["text"].strip()
            else:
                # Fallback: return the entire response as string
                logger.warning("Unknown response format, returning full response")
                logger.debug(f"Full response data: {response_data}")
                return str(response_data)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response text: {response_text[:200]}...")
            # Return raw response if JSON parsing fails
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return f"Error parsing response: {str(e)}"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate responses for the given prompts.
        This is the required abstract method implementation.
        """
        logger.info(f"Generating responses for {len(prompts)} prompt(s)")

        generations = []

        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing prompt {i+1}/{len(prompts)}")

                # Prepare payload
                payload = self._prepare_payload(prompt)

                # Add stop sequences if provided
                if stop:
                    payload["stop"] = stop

                logger.debug(f"Payload for prompt {i+1}: {json.dumps(payload)}")

                # Make curl request
                response_text = self._make_curl_request(payload)

                logger.debug(
                    f"Response Text: {response_text[:100]}..."
                )  # Log first 100 chars
                # Parse response
                generated_text = self._parse_response(response_text)

                # Apply stop sequences if they weren't handled by the API
                if stop and generated_text:
                    for stop_seq in stop:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]

                # Create Generation object
                generation = Generation(text=generated_text)
                generations.append([generation])
                logger.debug(f"Generated response length: {len(generated_text)}")

            except Exception as e:
                error_msg = f"Error generating response for prompt {i+1}: {str(e)}"
                logger.error(error_msg)
                # Create error generation
                error_generation = Generation(text=f"Error: {str(e)}")
                generations.append([error_generation])

        logger.info(f"Generated {len(generations)} response(s)")
        # Return LLMResult object
        return LLMResult(generations=generations)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM using curl."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Making curl LLM request to {self.api_url}")

        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,  # Ensure we get the complete response
            }
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

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
            logger.debug(f"Full curl command: {' '.join(curl_cmd)}")
            result = subprocess.run(
                curl_cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of failing
                timeout=self.timeout * 2
            )
            logger.debug(f"Curl exit code: {result.returncode}")
            logger.debug(f"Curl stderr: {result.stderr}")

            if result.returncode != 0:
                logger.error(
                    f"Curl command failed with return code {result.returncode}"
                )
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                raise Exception(f"Curl request failed: {result.stderr}")

            # Check if stdout is None (can happen with encoding issues)
            if result.stdout is None:
                logger.error("Curl command returned None output (possible encoding issue)")
                raise Exception("Curl command returned None output (encoding issue)")

            # Parse the response
            logger.debug(f"Raw curl response: {result.stdout[:500]}...")
            response_data = json.loads(result.stdout)
            logger.debug(
                f"Parsed response data: {json.dumps(response_data, indent=2)[:500]}..."
            )

            if "error" in response_data:
                raise Exception(f"API Error: {response_data['error']}")

            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                logger.debug(f"Choice data: {choice}")

                # Try different ways to extract content
                content = None
                if "message" in choice:
                    message = choice["message"]
                    # For DeepSeek R1: Always prioritize content over reasoning
                    if "content" in message and message["content"].strip():
                        content = message["content"]
                        # Log reasoning info if available
                        if "reasoning" in message and message["reasoning"]:
                            logger.debug(f"Response includes reasoning (length: {len(message['reasoning'])} chars)")
                    # Only use reasoning as fallback if content is truly empty
                    elif "reasoning" in message and message["reasoning"]:
                        content = message["reasoning"]
                        logger.warning(
                            "Content field is empty, using reasoning as fallback (may be incomplete response)"
                        )
                elif "text" in choice:
                    content = choice["text"]
                elif "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]

                if content is not None and content.strip():
                    logger.debug(f"Curl LLM response received (length: {len(content)})")
                    return content.strip()
                else:
                    logger.error(f"Could not extract content from choice: {choice}")
                    # Return the full choice as a fallback
                    return str(choice)
            else:
                logger.error(f"No choices in response: {response_data}")
                raise Exception("No valid response from API")

        except subprocess.TimeoutExpired:
            logger.error("Curl request timed out")
            raise Exception("Request timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response (first 1000 chars): {result.stdout[:1000]}")

            # Try to find and extract JSON from the response
            try:
                start_idx = result.stdout.find("{")
                if start_idx != -1:
                    json_part = result.stdout[start_idx:]
                    logger.debug(
                        f"Attempting to parse extracted JSON: {json_part[:200]}..."
                    )
                    response_data = json.loads(json_part)

                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        choice = response_data["choices"][0]
                        if "message" in choice:
                            message = choice["message"]
                            # Always prioritize content over reasoning
                            if "content" in message and message["content"].strip():
                                return message["content"]
                            elif "reasoning" in message and message["reasoning"]:
                                return message["reasoning"]

                    return str(response_data)
                else:
                    return result.stdout.strip()
            except:
                return result.stdout.strip()

            raise Exception(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Curl LLM request failed: {e}")
            raise

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream responses (optional implementation).
        Note: This basic implementation doesn't support true streaming via curl.
        """
        logger.info("Streaming not implemented for CurlLLM, using regular generation")

        # Generate the full response
        response = self._generate(
            [prompt], stop=stop, run_manager=run_manager, **kwargs
        )

        # Yield the response as a single chunk
        if response and response[0]:
            yield GenerationChunk(text=response[0][0])

    @classmethod
    def create_openai_compatible(
        cls,
        api_url: str,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        **kwargs,
    ) -> "CurlLLM":
        """
        Create a CurlLLM instance configured for OpenAI-compatible APIs.
        """
        return cls(api_url=api_url, api_key=api_key, model_name=model_name, **kwargs)

    @classmethod
    def create_custom_api(
        cls,
        api_url: str,
        api_key: Optional[str] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> "CurlLLM":
        """
        Create a CurlLLM instance for custom API endpoints.
        """
        return cls(api_url=api_url, api_key=api_key, headers=headers or {}, **kwargs)


# Example usage and testing
def test_curl_llm():
    """Test function for CurlLLM implementation."""
    logger.info("Testing CurlLLM implementation")

    curl_llm_url = (
        os.getenv("CURL_LLM_URL", "https://openrouter.ai/api/v1") + "/chat/completions"
    )
    curl_llm_key = os.getenv("CURL_LLM_API_KEY")
    curl_model_name = os.getenv("CURL_LLM_MODEL", "test-model")
    # Example configuration for a hypothetical API
    llm = CurlLLM(
        api_url=curl_llm_url,
        api_key=curl_llm_key,
        model_name=curl_model_name,
        temperature=0.7,
        max_tokens=2000,  # Increased to allow reasoning + content
        timeout=60,  # Increased timeout for longer responses
    )

    try:
        # Test the invoke method (public interface)
        test_prompt = "Hello, how are you?"
        response = llm.invoke(test_prompt)
        logger.info(f"Test response: {response}")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run test
    test_curl_llm()
