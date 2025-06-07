import logging
import os
from typing import Optional, Any
from langchain_google_genai import GoogleGenerativeAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class LLMFallbackHandler:
    """
    Handles LLM initialization with fallback mechanisms.
    Tries CurlLLM first, then falls back to Gemini or other configured LLMs.
    """

    @staticmethod
    def create_curl_llm(
        api_url: str,
        api_key: Optional[str] = None,
        model_name: str = "default-model",
        **kwargs,
    ) -> Optional[Any]:
        """
        Attempt to create a CurlLLM instance.
        Returns None if creation fails.
        """
        try:
            from curl_llm import CurlLLM  # Import your CurlLLM implementation

            logger.info(f"Attempting to create CurlLLM with URL: {api_url}")

            llm = CurlLLM(
                api_url=api_url, api_key=api_key, model_name=model_name, **kwargs
            )

            # Test the LLM with a simple query
            test_response = llm.invoke("Hello")
            logger.info("CurlLLM created and tested successfully")
            return llm

        except ImportError:
            logger.warning("CurlLLM class not found, skipping curl-based LLM")
            return None
        except NotImplementedError as e:
            logger.error(f"CurlLLM implementation incomplete: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create CurlLLM: {e}")
            return None

    @staticmethod
    def create_gemini_llm(api_key: SecretStr, **kwargs) -> Optional[GoogleGenerativeAI]:
        """
        Create a Google Gemini LLM instance.
        """
        try:
            logger.info("Creating Google Gemini LLM")

            llm = GoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-1.5-flash",
                temperature=kwargs.get("temperature", 0.7),
                **kwargs,
            )

            # Test the LLM
            test_response = llm.invoke("Hello")
            logger.info("Google Gemini LLM created and tested successfully")
            return llm

        except Exception as e:
            logger.error(f"Failed to create Google Gemini LLM: {e}")
            return None

    @staticmethod
    def get_primary_llm(**config) -> Any:
        """
        Get the primary LLM with fallback logic.

        Args:
            config: Configuration dictionary that may contain:
                - curl_api_url: URL for CurlLLM
                - curl_api_key: API key for CurlLLM
                - curl_model_name: Model name for CurlLLM
                - google_api_key: API key for Gemini
                - temperature: Temperature setting
                - Other LLM-specific parameters
        """
        logger.info("Initializing primary LLM with fallback logic")

        # Try CurlLLM first if configuration is available
        if config.get("curl_api_url"):
            logger.info("CurlLLM configuration found, attempting to use CurlLLM")

            curl_llm = LLMFallbackHandler.create_curl_llm(
                api_url=config["curl_api_url"],
                api_key=config.get("curl_api_key"),
                model_name=config.get("curl_model_name", "default-model"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1000),
                timeout=config.get("timeout", 30),
            )

            if curl_llm:
                logger.info("Successfully created CurlLLM")
                return curl_llm
            else:
                logger.warning("CurlLLM creation failed, falling back to Gemini")

        # Fallback to Google Gemini
        google_api_key = config.get("google_api_key")
        if google_api_key:
            logger.info("Falling back to Google Gemini LLM")

            gemini_llm = LLMFallbackHandler.create_gemini_llm(
                api_key=google_api_key, temperature=config.get("temperature", 0.7)
            )

            if gemini_llm:
                logger.info("Successfully created Google Gemini LLM")
                return gemini_llm
            else:
                logger.error("Google Gemini LLM creation failed")

        # If all else fails, raise an error
        logger.critical("No LLM could be initialized")
        raise RuntimeError(
            "Failed to initialize any LLM. Please check your configuration."
        )

    @staticmethod
    def get_env_config() -> dict:
        """
        Get LLM configuration from environment variables.
        """
        config = {}

        # CurlLLM configuration
        if os.getenv("CURL_API_URL"):
            config["curl_api_url"] = os.getenv("CURL_API_URL")
            config["curl_api_key"] = os.getenv("CURL_API_KEY")
            config["curl_model_name"] = os.getenv("CURL_MODEL_NAME", "default-model")

        # Google Gemini configuration
        if os.getenv("GOOGLE_API_KEY"):
            config["google_api_key"] = SecretStr(os.getenv("GOOGLE_API_KEY"))

        # General configuration
        config["temperature"] = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        config["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "1000"))
        config["timeout"] = int(os.getenv("LLM_TIMEOUT", "30"))

        return config


# Integration example for your main.py
def initialize_llm_with_fallback():
    """
    Example function showing how to integrate the fallback handler into your main.py
    """
    logger = logging.getLogger(__name__)

    try:
        # Get configuration from environment
        config = LLMFallbackHandler.get_env_config()

        # Or manually specify configuration
        # config = {
        #     "curl_api_url": "https://your-api.com/v1/completions",
        #     "curl_api_key": "your-curl-api-key",
        #     "curl_model_name": "your-model",
        #     "google_api_key": SecretStr("your-google-api-key"),
        #     "temperature": 0.7
        # }

        # Get the primary LLM
        llm = LLMFallbackHandler.get_primary_llm(**config)

        logger.info(f"Successfully initialized LLM: {type(llm).__name__}")
        return llm

    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        raise


# Modified section for your main.py
def get_llm_with_fallback():
    """
    Replace your existing LLM initialization code with this function.
    """
    logger = logging.getLogger(__name__)

    try:
        # Try the fallback handler approach
        return initialize_llm_with_fallback()

    except Exception as e:
        logger.warning(
            f"Fallback handler failed: {e}, using direct Gemini initialization"
        )

        # Direct fallback to your existing Gemini code
        api_key = SecretStr(os.getenv("GOOGLE_API_KEY"))
        if not api_key.get_secret_value():
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        llm = GoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.7,
        )

        # Test LLM connection
        test_response = llm.invoke("Hello, can you hear me?")
        logger.info("Direct Gemini LLM initialized successfully")
        return llm


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)

    # Test the fallback handler
    try:
        llm = initialize_llm_with_fallback()
        print(f"Successfully initialized: {type(llm).__name__}")

        # Test the LLM
        response = llm.invoke("What is the capital of France?")
        print(f"Test response: {response}")

    except Exception as e:
        print(f"Test failed: {e}")
