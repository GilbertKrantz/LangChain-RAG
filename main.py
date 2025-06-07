import logging
import os
import sys
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph
from langchain import hub
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import SecretStr
from utils.DocumentProcessor import (
    get_docs,
    get_docs_multiple,
    text_split,
    vectorize_docs,
    validate_docs,
    get_source_info,
)
from utils.LLMBuilder import LLMBuilder, State

# Remove IPython display for better compatibility
try:
    from IPython.display import Image, display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("IPython not available - graph visualization disabled")


# Configure logging
def setup_logging(quiet_mode=False):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/rag_app_{timestamp}.log"

    # Determine logging level from environment or parameter
    log_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
    if quiet_mode:
        log_level = "ERROR"

    # Convert string to logging level
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Configure handlers
    handlers = [logging.FileHandler(log_filename)]

    # Only add console handler if not in quiet mode
    if not quiet_mode and log_level != "CRITICAL":
        handlers.append(logging.StreamHandler(sys.stdout))

    # Configure logging format
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Get logger for this module
    logger = logging.getLogger(__name__)
    if not quiet_mode:
        logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


def get_document_sources():
    """Get document sources from environment variables or use defaults."""
    logger = logging.getLogger(__name__)

    # Check for document sources in environment variables
    doc_sources = os.getenv("DOCUMENT_SOURCES")
    doc_paths = os.getenv("DOCUMENT_PATHS")

    sources = []

    if doc_sources:
        # Parse comma-separated sources
        sources.extend([source.strip() for source in doc_sources.split(",")])
        logger.info(f"Found DOCUMENT_SOURCES: {sources}")

    if doc_paths:
        # Parse comma-separated local paths
        paths = [path.strip() for path in doc_paths.split(",")]
        sources.extend(paths)
        logger.info(f"Found DOCUMENT_PATHS: {paths}")

    if not sources:
        # Default to the original web source
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

            # Check if source exists (for local files/directories)
            if not source.startswith(("http://", "https://")):
                if not os.path.exists(source):
                    logger.error(f"Local path does not exist: {source}")
                    failed_sources.append((source, "Path does not exist"))
                    continue

            # Load documents from this source
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

    # Log results
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

    # Get source information
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
        # Build the graph
        graph_builder = StateGraph(State)

        # Add nodes in sequence
        node_names = ["analyze_query", "retrieve", "generate"]
        for i, node_name in enumerate(node_names):
            if node_name in kwargs:
                graph_builder.add_node(node_name, kwargs[node_name])
                logger.debug(f"Added node: {node_name}")
            else:
                logger.error(f"Missing required node function: {node_name}")
                raise ValueError(f"Missing required node function: {node_name}")

        # Add edges to create sequence
        graph_builder.add_edge(START, node_names[0])
        for i in range(len(node_names) - 1):
            graph_builder.add_edge(node_names[i], node_names[i + 1])

        # Compile the graph
        graph = graph_builder.compile()
        logger.info("Graph compilation completed successfully")

        # Display graph if IPython is available
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


def get_env_var(var_name):
    """Get environment variable, raise error if not set."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Retrieving environment variable: {var_name}")

    value = os.getenv(var_name)
    if value is None:
        logger.error(f"Environment variable '{var_name}' is not set")
        raise ValueError(f"Environment variable '{var_name}' is not set.")

    os.environ[var_name] = value  # Ensure it's set in the environment
    logger.info(f"Environment variable '{var_name}' retrieved successfully")
    return SecretStr(value)


def test_llm_connection(llm):
    """Test LLM connection with a simple query."""
    logger = logging.getLogger(__name__)
    logger.info("Testing LLM connection")

    try:
        test_response = llm.invoke("Hello, can you hear me?")
        logger.info("LLM connection test successful")

        # Handle different response types
        if hasattr(test_response, "content"):
            logger.debug(f"LLM test response: {test_response.content[:100]}...")
        elif isinstance(test_response, str):
            logger.debug(f"LLM test response: {test_response[:100]}...")
        else:
            logger.debug(f"LLM test response type: {type(test_response)}")

        return True
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False


def test_embeddings_connection(embeddings):
    """Test embeddings connection with a simple query."""
    logger = logging.getLogger(__name__)
    logger.info("Testing embeddings connection")

    try:
        test_embedding = embeddings.embed_query("test")
        logger.info(
            f"Embeddings connection test successful - dimension: {len(test_embedding)}"
        )
        return True
    except Exception as e:
        logger.error(f"Embeddings connection test failed: {e}")
        return False


def print_usage_info():
    """Print usage information for document sources."""
    print("\n" + "=" * 60)
    print("📖 DOCUMENT SOURCE CONFIGURATION")
    print("=" * 60)
    print("You can specify document sources using environment variables:")
    print()
    print("🌐 DOCUMENT_SOURCES - Web URLs (comma-separated)")
    print(
        "   Example: DOCUMENT_SOURCES='https://example.com/page1,https://example.com/page2'"
    )
    print()
    print("📁 DOCUMENT_PATHS - Local files/directories (comma-separated)")
    print("   Example: DOCUMENT_PATHS='/path/to/file.txt,/path/to/directory'")
    print()
    print("💡 Supported file types: .txt, .md, .py, .js, .html, .css, .json, .xml")
    print("💡 For directories, all supported files will be loaded recursively")
    print("💡 You can mix web URLs and local paths in the same session")
    print()
    print("🔇 QUIET MODE OPTIONS:")
    print("   - Use --quiet or -q flag: uv run main.py --quiet")
    print("   - Set environment: QUIET_MODE=true uv run main.py")
    print("   - Set log level: LOGGING_LEVEL=ERROR uv run main.py")
    print("=" * 60)


def main():
    # Check for quiet mode
    quiet_mode = (
        "--quiet" in sys.argv
        or "-q" in sys.argv
        or os.getenv("QUIET_MODE", "").lower() in ["true", "1", "yes"]
    )

    # Initialize logging
    logger = setup_logging(quiet_mode=quiet_mode)
    if not quiet_mode:
        logger.info("=== Enhanced RAG Application Starting ===")

    question_count = 0  # Initialize here to avoid UnboundLocalError

    try:
        # Print usage information (only if not in quiet mode)
        if not quiet_mode:
            print_usage_info()

        # Check for required environment variables first
        logger.info("Checking environment variables")
        try:
            api_key = get_env_var("GOOGLE_API_KEY")
        except ValueError as e:
            logger.error(f"Environment setup failed: {e}")
            print(f"Setup Error: {e}")
            print("Please set the GOOGLE_API_KEY environment variable.")
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
                "Could not retrieve RAG prompt. Please check your internet connection and LangSmith access."
            )
            return 1

        # Initialize LLM
        logger.info("Initializing Google Gemini LLM")
        try:
            llm = GoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-1.5-flash",  # or "gemini-1.5-pro" for better performance
                temperature=0.7,
            )
            logger.info("Google Gemini LLM initialized successfully")

            # Test LLM connection
            if not test_llm_connection(llm):
                logger.error("LLM connection test failed")
                print(
                    "Error: Could not connect to Google Gemini LLM. Please check your API key and internet connection."
                )
                return 1

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            print(f"LLM Error: {e}")
            return 1

        # Initialize embeddings
        logger.info("Initializing Google Gemini embeddings")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=api_key, model="models/embedding-001"
            )
            logger.info("Google Gemini embeddings initialized successfully")

            # Test embeddings connection
            if not test_embeddings_connection(embeddings):
                logger.error("Embeddings connection test failed")
                print(
                    "Error: Could not connect to Google Gemini embeddings. Please check your API key and internet connection."
                )
                return 1

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            print(f"Embeddings Error: {e}")
            return 1

        # Initialize vector store
        logger.info("Initializing InMemory vector store with Gemini embeddings")
        try:
            vector_store = InMemoryVectorStore(embeddings)
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            print(f"Vector Store Error: {e}")
            return 1

        # Get document sources
        logger.info("Determining document sources")
        sources = get_document_sources()

        # Load and process documents
        logger.info("Loading documents from configured sources")
        try:
            docs, successful_sources, failed_sources = load_documents_from_sources(
                sources
            )

            if not docs:
                logger.error("No documents were loaded from any source")
                print("❌ Error: No documents could be loaded from any source.")
                if failed_sources:
                    print("\nFailed sources:")
                    for source, error in failed_sources:
                        print(f"  - {source}: {error}")
                return 1

            # Display source summary
            display_source_summary(docs)

            # Validate documents
            logger.info("Validating loaded documents")
            docs = validate_docs(docs)

            if not docs:
                logger.error("No valid documents after validation")
                print("❌ Error: No valid documents found after validation.")
                return 1

        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            print(f"Document Loading Error: {e}")
            return 1

        logger.info("Splitting documents into texts")
        try:
            texts = text_split(docs)
            logger.info(f"Split into {len(texts)} text chunks")

            if len(texts) == 0:
                logger.warning("No text chunks created")
                print("Warning: No text chunks were created from documents.")

        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            print(f"Document Splitting Error: {e}")
            return 1

        logger.info("Vectorizing documents")
        try:
            document_ids = vectorize_docs(texts, vector_store)
            logger.info(f"Vectorized documents with {len(document_ids)} document IDs")

            if len(document_ids) == 0:
                logger.warning("No documents vectorized")
                print("Warning: No documents were successfully vectorized.")

        except Exception as e:
            logger.error(f"Failed to vectorize documents: {e}")
            print(f"Document Vectorization Error: {e}")
            return 1

        # Build LLM with configuration
        logger.info("Building LLM with configuration")
        try:
            llm_builder = LLMBuilder.from_config(
                {"vector_store": vector_store, "prompt": prompt, "llm": llm}
            )
            logger.info("LLM builder configured successfully")

            # Get stats for verification
            stats = llm_builder.get_stats()
            logger.info(f"LLM Builder Stats: {stats}")

        except Exception as e:
            logger.error(f"Failed to build LLM configuration: {e}")
            print(f"LLM Builder Error: {e}")
            return 1

        # Get methods from LLM builder
        retrieve = llm_builder.retrieve
        generate = llm_builder.generate
        analyze_query = llm_builder.analyze_query

        # Compile graph
        logger.info("Compiling application graph")
        try:
            graph = compiler(
                analyze_query=analyze_query, retrieve=retrieve, generate=generate
            )
            logger.info("Application setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to compile graph: {e}")
            print(f"Graph Compilation Error: {e}")
            return 1

        # Interactive loop
        print("\n" + "=" * 50)
        print("🚀 Enhanced RAG Application started successfully!")
        print("📚 Documents loaded and indexed from multiple sources")
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

                # Show some stats if context was found
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
