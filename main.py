import logging
import os
import sys
import json
import subprocess
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph
from langchain import hub
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from langchain_core.outputs import LLMResult, Generation
from pydantic import SecretStr
from typing import List, Optional, Any, Dict
from utils.DocumentProcessor import (
    get_docs,
    get_docs_multiple,
    text_split,
    vectorize_docs,
    validate_docs,
    get_source_info,
)
from utils.LLM.LLMBuilder import LLMBuilder, State
from utils.LLM.CUrlLLMBuilder import CurlLLM
from utils.Embeddings.CUrlEmbeddings import CurlEmbeddings

# Remove IPython display for better compatibility
try:
    from IPython.display import Image, display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("IPython not available - graph visualization disabled")


def setup_logging(quiet_mode=False):
    """Set up logging configuration."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/rag_app_{timestamp}.log"

    log_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
    if quiet_mode:
        log_level = "ERROR"

    numeric_level = getattr(logging, log_level, logging.INFO)
    handlers = [logging.FileHandler(log_filename)]

    if not quiet_mode and log_level != "CRITICAL":
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(__name__)
    if not quiet_mode:
        logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


def create_llm_and_embeddings():
    """Create LLM and embeddings with curl as primary and Gemini as fallback."""
    logger = logging.getLogger(__name__)

    # Check for curl configuration
    curl_llm_url = os.getenv("CURL_LLM_URL")
    curl_llm_key = os.getenv("CURL_LLM_API_KEY")
    curl_embedding_url = os.getenv("CURL_EMBEDDING_URL")
    curl_embedding_key = os.getenv("CURL_EMBEDDING_API_KEY")

    # Gemini fallback
    gemini_key = os.getenv("GOOGLE_API_KEY")

    llm = None
    embeddings = None

    # Try to create curl-based LLM
    if curl_llm_url and curl_llm_key:
        try:
            logger.info("Attempting to create curl-based LLM")
            curl_model = os.getenv("CURL_LLM_MODEL", "gpt-3.5-turbo")

            # Add /chat/completions endpoint if not already present
            if not curl_llm_url.endswith("/chat/completions"):
                curl_llm_url = curl_llm_url.rstrip("/") + "/chat/completions"

            llm = CurlLLM(
                api_url=curl_llm_url,
                api_key=curl_llm_key,
                model_name=curl_model,
                temperature=0.7,
                max_tokens=2000,  # Increased for DeepSeek R1 reasoning + content
                timeout=60,  # Increased timeout for longer responses
            )

            # Test curl LLM using invoke() method
            test_response = llm.invoke("Hello, this is a test.")
            logger.info(f"Curl LLM test successful: {test_response[:100]}...")

        except Exception as e:
            logger.warning(f"Curl LLM failed: {e}, falling back to Gemini")
            llm = None

    # Try to create curl-based embeddings
    if curl_embedding_url and curl_embedding_key:
        try:
            logger.info("Attempting to create curl-based embeddings")
            curl_embed_model = os.getenv(
                "CURL_EMBEDDING_MODEL", "text-embedding-ada-002"
            )
            embeddings = CurlEmbeddings(
                api_url=curl_embedding_url,
                api_key=curl_embedding_key,
                model_name=curl_embed_model,
            )

            # Test curl embeddings
            test_embedding = embeddings.embed_query("test")
            if test_embedding:
                logger.info("Curl embeddings test successful")
            else:
                raise Exception("Empty embedding returned")

        except Exception as e:
            logger.warning(f"Curl embeddings failed: {e}, falling back to Gemini")
            embeddings = None

    # Fallback to Gemini if needed
    if not llm and gemini_key:
        try:
            logger.info("Creating Gemini LLM as fallback")
            llm = GoogleGenerativeAI(
                google_api_key=SecretStr(gemini_key),
                model="gemini-1.5-flash",
                temperature=0.7,
            )
            test_response = llm.invoke("Hello, this is a test.")
            logger.info(f"Gemini LLM fallback successful: {test_response[:100]}...")
        except Exception as e:
            logger.error(f"Gemini LLM fallback failed: {e}")
            raise Exception("No working LLM available")

    if not embeddings and gemini_key:
        try:
            logger.info("Creating Gemini embeddings as fallback")
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=SecretStr(gemini_key), model="models/embedding-001"
            )
            test_embedding = embeddings.embed_query("test")
            logger.info("Gemini embeddings fallback successful")
        except Exception as e:
            logger.error(f"Gemini embeddings fallback failed: {e}")
            raise Exception("No working embeddings available")

    if not llm:
        raise Exception("No LLM could be initialized")
    if not embeddings:
        raise Exception("No embeddings could be initialized")

    return llm, embeddings


def get_document_sources():
    """Get document sources from environment variables or use defaults."""
    logger = logging.getLogger(__name__)

    doc_sources = os.getenv("DOCUMENT_SOURCES")
    doc_paths = os.getenv("DOCUMENT_PATHS")

    sources = []

    if doc_sources:
        sources.extend([source.strip() for source in doc_sources.split(",")])
        logger.info(f"Found DOCUMENT_SOURCES: {sources}")

    if doc_paths:
        paths = [path.strip() for path in doc_paths.split(",")]
        sources.extend(paths)
        logger.info(f"Found DOCUMENT_PATHS: {paths}")

    if not sources:
        default_source = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        sources.append(default_source)
        logger.info(f"No custom sources found, using default: {default_source}")

    return sources


def load_documents_from_sources(sources):
    """Load documents from multiple sources with comprehensive error handling."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading documents from {len(sources)} source(s)")

    all_docs = []
    successful_sources = []
    failed_sources = []

    for i, source in enumerate(sources):
        try:
            logger.info(f"Loading source {i+1}/{len(sources)}: {source}")

            if not source.startswith(("http://", "https://")):
                if not os.path.exists(source):
                    logger.error(f"Local path does not exist: {source}")
                    failed_sources.append((source, "Path does not exist"))
                    continue

            docs = get_docs(source)

            if docs:
                all_docs.extend(docs)
                successful_sources.append(source)
                logger.info(f"Successfully loaded {len(docs)} documents from: {source}")
            else:
                logger.warning(f"No documents loaded from: {source}")
                failed_sources.append((source, "No documents found"))

        except Exception as e:
            logger.error(f"Failed to load from {source}: {e}")
            failed_sources.append((source, str(e)))
            continue

    logger.info(f"Document loading complete:")
    logger.info(f"  - Successful sources: {len(successful_sources)}")
    logger.info(f"  - Failed sources: {len(failed_sources)}")
    logger.info(f"  - Total documents: {len(all_docs)}")

    if failed_sources:
        logger.warning("Failed sources:")
        for source, error in failed_sources:
            logger.warning(f"  - {source}: {error}")

    return all_docs, successful_sources, failed_sources


def display_source_summary(docs):
    """Display a summary of loaded document sources."""
    logger = logging.getLogger(__name__)

    if not docs:
        print("⚠️  No documents loaded!")
        return

    source_info = get_source_info(docs)

    print(f"\n📚 Document Summary:")
    print(f"  📄 Total documents: {source_info['total']}")

    if source_info["web"] > 0:
        print(f"  🌐 Web documents: {source_info['web']}")

    if source_info["file"] > 0:
        print(f"  📁 File documents: {source_info['file']}")

    if source_info["directory"] > 0:
        print(f"  📂 Directory documents: {source_info['directory']}")

    if source_info["file_types"]:
        print(f"  📋 File types: {dict(source_info['file_types'])}")

    logger.info(f"Document source summary displayed: {source_info}")


def compiler(**kwargs):
    """Compile application and test"""
    logger = logging.getLogger(__name__)
    logger.info("Starting graph compilation")

    try:
        graph_builder = StateGraph(State)

        node_names = ["analyze_query", "retrieve", "generate"]
        for i, node_name in enumerate(node_names):
            if node_name in kwargs:
                graph_builder.add_node(node_name, kwargs[node_name])
                logger.debug(f"Added node: {node_name}")
            else:
                logger.error(f"Missing required node function: {node_name}")
                raise ValueError(f"Missing required node function: {node_name}")

        graph_builder.add_edge(START, node_names[0])
        for i in range(len(node_names) - 1):
            graph_builder.add_edge(node_names[i], node_names[i + 1])

        graph = graph_builder.compile()
        logger.info("Graph compilation completed successfully")

        if IPYTHON_AVAILABLE:
            try:
                display(Image(graph.get_graph().draw_mermaid_png()))
                logger.debug("Graph visualization displayed")
            except Exception as e:
                logger.warning(f"Could not display graph visualization: {e}")

        return graph

    except Exception as e:
        logger.error(f"Error during graph compilation: {e}")
        raise


def invoke(graph, question):
    """Invoke the graph with a question."""
    logger = logging.getLogger(__name__)
    logger.info(
        f"Invoking graph with question: {question[:100]}{'...' if len(question) > 100 else ''}"
    )

    try:
        state = {"question": question}
        response = graph.invoke(state)
        logger.info("Graph invocation completed successfully")
        logger.debug(
            f"Response preview: {str(response)[:200]}{'...' if len(str(response)) > 200 else ''}"
        )
        return response
    except Exception as e:
        logger.error(f"Error during graph invocation: {e}")
        raise


def print_usage_info():
    """Print usage information for document sources and curl configuration."""
    print("\n" + "=" * 60)
    print("📖 RAG APPLICATION CONFIGURATION")
    print("=" * 60)
    print("🔧 CURL CONFIGURATION (Primary):")
    print(
        "   CURL_LLM_URL - API endpoint for LLM requests (e.g., https://openrouter.ai/api/v1)"
    )
    print("   CURL_LLM_API_KEY - API key for LLM")
    print(
        "   CURL_LLM_MODEL - Model name (e.g., deepseek/deepseek-r1-0528-qwen3-8b:free)"
    )
    print("   CURL_EMBEDDING_URL - API endpoint for embeddings")
    print("   CURL_EMBEDDING_API_KEY - API key for embeddings")
    print("   CURL_EMBEDDING_MODEL - Embedding model (default: text-embedding-ada-002)")
    print(
        "   Note: /chat/completions will be automatically added to CURL_LLM_URL if not present"
    )
    print()
    print("🔄 GEMINI FALLBACK:")
    print("   GOOGLE_API_KEY - Google Gemini API key (fallback)")
    print()
    print("📁 DOCUMENT SOURCES:")
    print("   DOCUMENT_SOURCES - Web URLs (comma-separated)")
    print("   DOCUMENT_PATHS - Local files/directories (comma-separated)")
    print()
    print("🔇 QUIET MODE OPTIONS:")
    print("   --quiet or -q flag: python main.py --quiet")
    print("   QUIET_MODE=true python main.py")
    print("   LOGGING_LEVEL=ERROR python main.py")
    print("=" * 60)


def main():
    quiet_mode = (
        "--quiet" in sys.argv
        or "-q" in sys.argv
        or os.getenv("QUIET_MODE", "").lower() in ["true", "1", "yes"]
    )

    logger = setup_logging(quiet_mode=quiet_mode)
    if not quiet_mode:
        logger.info("=== Enhanced RAG Application with Curl Support Starting ===")

    question_count = 0

    try:
        if not quiet_mode:
            print_usage_info()

        # Initialize LLM and embeddings with curl primary, Gemini fallback
        logger.info("Initializing LLM and embeddings")
        try:
            llm, embeddings = create_llm_and_embeddings()
            logger.info("LLM and embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM/embeddings: {e}")
            print(f"Initialization Error: {e}")
            print("Please check your API keys and configuration.")
            return 1

        # Pull RAG prompt from LangSmith Hub
        logger.info("Pulling RAG prompt from LangSmith Hub")
        try:
            prompt = hub.pull("rlm/rag-prompt")
            logger.info("RAG prompt retrieved successfully")
        except Exception as e:
            logger.error(f"Failed to pull prompt from hub: {e}")
            print(f"Prompt Error: {e}")
            print(
                "Could not retrieve RAG prompt. Please check your internet connection."
            )
            return 1

        # Initialize vector store
        logger.info("Initializing vector store")
        try:
            vector_store = InMemoryVectorStore(embeddings)
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            print(f"Vector Store Error: {e}")
            return 1

        # Get document sources and load documents
        logger.info("Loading documents")
        sources = get_document_sources()
        docs, successful_sources, failed_sources = load_documents_from_sources(sources)

        if not docs:
            logger.error("No documents were loaded")
            print("❌ Error: No documents could be loaded from any source.")
            return 1

        display_source_summary(docs)
        docs = validate_docs(docs)

        if not docs:
            logger.error("No valid documents after validation")
            print("❌ Error: No valid documents found after validation.")
            return 1

        # Process documents
        logger.info("Processing documents")
        texts = text_split(docs)
        document_ids = vectorize_docs(texts, vector_store)

        # Build LLM configuration
        logger.info("Building LLM configuration")
        llm_builder = LLMBuilder.from_config(
            {"vector_store": vector_store, "prompt": prompt, "llm": llm}
        )

        # Compile graph
        logger.info("Compiling application graph")
        graph = compiler(
            analyze_query=llm_builder.analyze_query,
            retrieve=llm_builder.retrieve,
            generate=llm_builder.generate,
        )

        # Interactive loop
        print("\n" + "=" * 50)
        print("🚀 Enhanced RAG Application with Curl Support ready!")
        print("📚 Documents loaded and indexed")
        print("🤖 AI assistant ready to answer your questions")
        print("Type 'exit', 'quit', or 'q' to quit.")
        print("Type 'help' for usage information.")
        print("=" * 50)

        logger.info("Starting interactive loop")

        while True:
            try:
                question = input("\n💭 Enter your question: ")

                if question.lower() in ["exit", "quit", "q"]:
                    logger.info("User requested exit")
                    print("👋 Goodbye!")
                    break

                if question.lower() in ["help", "h"]:
                    print_usage_info()
                    continue

                if not question.strip():
                    logger.warning("Empty question received")
                    print("⚠️  Please enter a valid question.")
                    continue

                question_count += 1
                logger.info(f"Processing question #{question_count}")

                print("🔍 Searching for relevant information...")
                response = invoke(graph, question)

                print(f"\n🤖 Answer: {response.get('answer', 'No answer generated')}")

                context_docs = response.get("context", [])
                if context_docs:
                    print(f"📄 Based on {len(context_docs)} relevant document(s)")

                logger.info(f"Question #{question_count} processed successfully")

            except KeyboardInterrupt:
                logger.info("Application interrupted by user (Ctrl+C)")
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(
                    f"Error processing question #{question_count}: {e}", exc_info=True
                )
                print(f"❌ Error: {e}")
                print("🔄 Please try again with a different question.")

    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        print(f"💥 Critical error: {e}")
        return 1

    finally:
        logger.info("=== Enhanced RAG Application Shutting Down ===")
        logger.info(f"Total questions processed: {question_count}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
