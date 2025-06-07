import logging
import os
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph
from langchain import hub
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import SecretStr
from utils.DocumentProcessor import get_docs, text_split, vectorize_docs
from utils.LLMBuilder import LLMBuilder, State

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/rag_app_{timestamp}.log'
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

def compiler(retrieve, generate):
    """Compile application and test"""
    logger = logging.getLogger(__name__)
    logger.info("Starting graph compilation")
    
    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        logger.info("Graph compilation completed successfully")
        return graph
    except Exception as e:
        logger.error(f"Error during graph compilation: {e}")
        raise

def invoke(graph, question):
    """Invoke the graph with a question."""
    logger = logging.getLogger(__name__)
    logger.info(f"Invoking graph with question: {question[:100]}{'...' if len(question) > 100 else ''}")
    
    try:
        state = {"question": question}
        response = graph.invoke(state)
        logger.info("Graph invocation completed successfully")
        logger.debug(f"Response preview: {str(response)[:200]}{'...' if len(str(response)) > 200 else ''}")
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

def main():
    # Initialize logging
    logger = setup_logging()
    logger.info("=== RAG Application Starting ===")
    
    try:
        # Pull RAG prompt from LangSmith Hub
        logger.info("Pulling RAG prompt from LangSmith Hub")
        prompt = hub.pull("rlm/rag-prompt")
        logger.info("RAG prompt retrieved successfully")
        
        # Initialize LLM
        logger.info("Initializing Google Gemini LLM")
        llm = GoogleGenerativeAI(
            google_api_key=get_env_var("GOOGLE_API_KEY"),
            model="gemini-1.5-flash",  # or "gemini-1.5-pro" for better performance
            temperature=0.7,
        )
        logger.info("Google Gemini LLM initialized successfully")
        
        # Initialize embeddings
        logger.info("Initializing Google Gemini embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=get_env_var("GOOGLE_API_KEY"),
            model="models/embedding-001"
        )
        logger.info("Google Gemini embeddings initialized successfully")
        
        # Initialize vector store
        logger.info("Initializing InMemory vector store with Gemini embeddings")
        vector_store = InMemoryVectorStore(embeddings)
        logger.info("Vector store initialized successfully")
        
        # Load and process documents
        logger.info("Loading documents")
        docs = get_docs()
        logger.info(f"Loaded {len(docs)} documents")
        
        logger.info("Splitting documents into texts")
        texts = text_split(docs)
        logger.info(f"Split into {len(texts)} text chunks")
        
        logger.info("Vectorizing documents")
        document_ids = vectorize_docs(texts, vector_store)
        logger.info(f"Vectorized documents with {len(document_ids)} document IDs")
        
        # Build LLM with configuration
        logger.info("Building LLM with configuration")
        llm_builder = LLMBuilder.from_config({
            "vector_store": vector_store,
            "prompt": prompt,
            "llm": llm
        })
        logger.info("LLM builder configured successfully")
        
        retrieve = llm_builder.retrieve
        generate = llm_builder.generate
        
        # Compile graph
        graph = compiler(retrieve, generate)
        logger.info("Application setup completed successfully")
        
        # Interactive loop
        print("RAG Application started. Type 'exit' to quit.")
        logger.info("Starting interactive loop")
        
        question_count = 0
        while True:
            try:
                question = input("\nEnter your question: ")
                
                if question.lower() in ["exit", "quit", "q"]:
                    logger.info("User requested exit")
                    print("Goodbye!")
                    break
                
                if not question.strip():
                    logger.warning("Empty question received")
                    print("Please enter a valid question.")
                    continue
                
                question_count += 1
                logger.info(f"Processing question #{question_count}")
                
                response = invoke(graph, question)
                print(f"\nAnswer: {response['answer']}")
                logger.info(f"Question #{question_count} processed successfully")
                
            except KeyboardInterrupt:
                logger.info("Application interrupted by user (Ctrl+C)")
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing question #{question_count}: {e}", exc_info=True)
                print(f"Error: {e}")
                print("Please try again.")
                
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        print(f"Critical error: {e}")
        return 1
    
    finally:
        logger.info("=== RAG Application Shutting Down ===")
        logger.info(f"Total questions processed: {question_count if 'question_count' in locals() else 0}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)