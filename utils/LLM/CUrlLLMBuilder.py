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
    api_url: str = Field(..., description="The API endpoint URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    model_name: str = Field("curl-model", description="Model name identifier")
    temperature: float = Field(0.7, description="Temperature for response generation")
    max_tokens: int = Field(2000, description="Maximum tokens in response")
    timeout: int = Field(30, description="Request timeout in seconds")
    headers: dict = Field(default_factory=dict, description="Additional HTTP headers")
    show_reasoning: bool = Field(
        False, description="Whether to include reasoning in output"
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "curl_llm"

    def _prepare_headers(self) -> dict:
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"
        default_headers.update(self.headers)
        return default_headers

    def _prepare_payload(self, prompt: str, stream: bool = False) -> dict:
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    def _make_curl_request(self, payload: dict, stream: bool = False) -> Any:
        headers = self._prepare_headers()
        curl_cmd = [
            "curl",
            "-s",
            "-X",
            "POST",
            "--ssl-no-revoke",
            "--tlsv1.2",
            "--max-time",
            str(self.timeout),
        ]
        for key, value in headers.items():
            curl_cmd.extend(["-H", f"{key}: {value}"])
        curl_cmd.extend(["-d", json.dumps(payload), self.api_url])
        logger.debug(f"Executing curl command: {' '.join(curl_cmd[:10])}...")
        try:
            if stream:
                process = subprocess.Popen(
                    curl_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                return process
            else:
                result = subprocess.run(
                    curl_cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout + 5,
                )
                if result.returncode != 0:
                    error_msg = f"Curl command failed with return code {result.returncode}: {result.stderr}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                if result.stdout is None:
                    error_msg = (
                        "Curl command returned None output (possible encoding issue)"
                    )
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

    def _parse_response(self, response_text: str) -> dict:
        try:
            cleaned_response = response_text.strip()
            start_idx = cleaned_response.find("{")
            if start_idx != -1:
                cleaned_response = cleaned_response[start_idx:]
            response_data = json.loads(cleaned_response)
            logger.debug(
                f"Parsed response data: {json.dumps(response_data, indent=2)[:200]}..."
            )
            result = {"content": "", "reasoning": ""}
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                if "message" in choice:
                    message = choice["message"]
                    if "content" in message and message["content"]:
                        result["content"] = message["content"].strip()
                        if (
                            "reasoning" in message
                            and message["reasoning"]
                            and self.show_reasoning
                        ):
                            result["reasoning"] = message["reasoning"].strip()
                    elif "reasoning" in message and message["reasoning"]:
                        result["content"] = message["reasoning"].strip()
                        logger.warning(
                            "Content field empty, using reasoning as fallback"
                        )
                    elif "text" in choice:
                        result["content"] = choice["text"].strip()
                    else:
                        result["content"] = str(choice).strip()
                else:
                    result["content"] = str(choice).strip()
            elif "response" in response_data:
                result["content"] = response_data["response"].strip()
            elif "content" in response_data:
                result["content"] = response_data["content"].strip()
            elif "text" in response_data:
                result["content"] = response_data["text"].strip()
            else:
                logger.warning("Unknown response format")
                result["content"] = str(response_data)
                raise Exception("Unknown response format, no content found")
            if not result["content"]:
                logger.warning("Empty content in response, using raw response")
                result["content"] = cleaned_response.strip()
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"content": response_text.strip(), "reasoning": ""}
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {"content": f"Error parsing response: {str(e)}", "reasoning": ""}

    def _parse_stream_response(
        self, stream_process: subprocess.Popen
    ) -> Iterator[dict]:
        try:
            for line in stream_process.stdout:
                if line.strip():
                    logger.debug(f"Stream line received: {line[:100]}...")
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            result = {"content": "", "reasoning": ""}
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                if "delta" in choice:
                                    if (
                                        "content" in choice["delta"]
                                        and choice["delta"]["content"]
                                    ):
                                        result["content"] = choice["delta"]["content"]
                                    if (
                                        "reasoning" in choice["delta"]
                                        and self.show_reasoning
                                        and choice["delta"]["reasoning"]
                                    ):
                                        result["reasoning"] = choice["delta"][
                                            "reasoning"
                                        ]
                            yield result
                        except json.JSONDecodeError:
                            logger.debug(
                                f"Skipping non-JSON stream line: {line[:100]}..."
                            )
                            continue
            stderr = stream_process.stderr.read()
            stream_process.wait()
            if stream_process.returncode != 0:
                logger.error(
                    f"Stream failed with return code {stream_process.returncode}: {stderr}"
                )
                raise RuntimeError(f"Stream failed: {stderr}")
        except Exception as e:
            logger.error(f"Error in stream parsing: {e}")
            raise

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        logger.info(f"Generating responses for {len(prompts)} prompt(s)")
        generations = []
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                payload = self._prepare_payload(prompt)
                if stop:
                    payload["stop"] = stop
                logger.debug(f"Payload for prompt {i+1}: {json.dumps(payload)}")
                response_text = self._make_curl_request(payload)
                generated = self._parse_response(response_text)
                generation = Generation(
                    text=generated["content"],
                    generation_info=(
                        {"reasoning": generated["reasoning"]}
                        if generated["reasoning"]
                        else None
                    ),
                )
                generations.append([generation])
                if (
                    generated["content"]
                    == "Error parsing response: Unknown response format, no content found"
                ):
                    logger.warning(
                        f"Prompt {i+1} returned unknown response format, using raw response"
                    )
                    raise Exception("Unknown response format, no content found")

                logger.debug(f"Generated response length: {len(generated['content'])}")
                logger.info(f"Generated {len(generations)} response(s)")
                return LLMResult(generations=generations)
            except Exception as e:
                error_msg = f"Error generating response for prompt {i+1}: {str(e)}"
                logger.error(error_msg)
                generations.append([Generation(text=f"Error: {str(e)}")])

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        logger.debug(f"Making curl LLM request to {self.api_url}")
        try:
            payload = self._prepare_payload(prompt)
            if stop:
                payload["stop"] = stop
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            response_text = self._make_curl_request(payload)
            response = self._parse_response(response_text)
            content = response["content"]
            if response["reasoning"] and self.show_reasoning:
                logger.debug(
                    f"Reasoning included (length: {len(response['reasoning'])})"
                )
            if len(content) > self.max_tokens:
                logger.warning(
                    f"Response content exceeds max tokens ({len(content)} > {self.max_tokens})"
                )
            return content.strip()
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
        logger.info(f"Streaming response for prompt: {prompt[:100]}...")
        try:
            payload = self._prepare_payload(prompt, stream=True)
            if stop:
                payload["stop"] = stop
            stream_process = self._make_curl_request(payload, stream=True)
            for chunk in self._parse_stream_response(stream_process):
                yield GenerationChunk(
                    text=chunk["content"],
                    generation_info=(
                        {"reasoning": chunk["reasoning"]}
                        if chunk["reasoning"]
                        else None
                    ),
                )
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    @classmethod
    def create_openai_compatible(
        cls,
        api_url: str,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        **kwargs,
    ) -> "CurlLLM":
        return cls(api_url=api_url, api_key=api_key, model_name=model_name, **kwargs)

    @classmethod
    def create_custom_api(
        cls,
        api_url: str,
        api_key: Optional[str] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> "CurlLLM":
        return cls(api_url=api_url, api_key=api_key, headers=headers or {}, **kwargs)


def test_curl_llm():
    logger.info("Testing CurlLLM implementation")
    curl_llm_url = (
        os.getenv("CURL_LLM_URL", "https://openrouter.ai/api/v1") + "/chat/completions"
    )
    curl_llm_key = os.getenv("CURL_LLM_API_KEY")
    curl_model_name = os.getenv("CURL_LLM_MODEL", "test-model")
    llm = CurlLLM(
        api_url=curl_llm_url,
        api_key=curl_llm_key,
        model_name=curl_model_name,
        temperature=0.7,
        max_tokens=2000,
        timeout=60,
        show_reasoning=True,
    )
    try:
        test_prompt = "Hello, how are you?"
        response = llm.invoke(test_prompt)
        logger.info(f"Test response: {response}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    test_curl_llm()
