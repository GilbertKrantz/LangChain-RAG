import logging
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# Get logger for this module
logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class LLMBuilder:
    def __init__(self, vector_store, prompt, llm):
        self.vector_store = vector_store
        self.prompt = prompt
        self.llm = llm
        logger.info("LLMBuilder initialized successfully")
        
    @classmethod
    def from_config(cls, config):
        """Create LLMBuilder from configuration dictionary."""
        logger.info("Creating LLMBuilder from configuration")
        
        try:
            vector_store = config["vector_store"]
            prompt = config["prompt"]
            llm = config["llm"]
            
            logger.info("Configuration validated successfully")
            return cls(vector_store, prompt, llm)
            
        except KeyError as e:
            logger.error(f"Missing required configuration key: {e}")
            raise ValueError(f"Missing required configuration key: {e}")
        except Exception as e:
            logger.error(f"Error creating LLMBuilder from config: {e}")
            raise
    
    def retrieve(self, state: State):
        """Retrieve relevant documents based on the question."""
        question = state["question"]
        logger.info(f"Retrieving documents for question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Perform similarity search
            retrieved_docs = self.vector_store.similarity_search(question)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            logger.debug(f"Document lengths: {[len(doc.page_content) for doc in retrieved_docs]}")
            
            return {"context": retrieved_docs}
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            # Return empty context to allow the chain to continue
            return {"context": []}
    
    def generate(self, state: State):
        """Generate answer based on retrieved context and question."""
        question = state["question"]
        context_docs = state.get("context", [])
        
        logger.info(f"Generating answer for question with {len(context_docs)} context documents")
        
        try:
            # Prepare context from retrieved documents
            if context_docs:
                docs_content = "\n\n".join(doc.page_content for doc in context_docs)
                logger.debug(f"Context length: {len(docs_content)} characters")
            else:
                docs_content = "No relevant context found."
                logger.warning("No context documents available for generation")
            
            # Invoke the prompt with question and context
            messages = self.prompt.invoke({
                "question": question, 
                "context": docs_content
            })
            
            logger.debug(f"Prompt prepared, invoking LLM")
            
            # Invoke the LLM
            response = self.llm.invoke(messages)
            
            # Handle different response types
            if hasattr(response, 'content'):
                # For chat models that return objects with content attribute
                answer = response.content
                logger.debug("Response extracted from .content attribute")
            elif isinstance(response, str):
                # For models that return strings directly (like Gemini)
                answer = response
                logger.debug("Response is direct string")
            else:
                # Fallback: convert to string
                answer = str(response)
                logger.warning(f"Unexpected response type: {type(response)}, converted to string")
            
            logger.info(f"Answer generated successfully (length: {len(answer)} characters)")
            return {"answer": answer}
            
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            # Return a fallback answer
            fallback_answer = f"I apologize, but I encountered an error while generating an answer: {str(e)}"
            return {"answer": fallback_answer}
    
    def retrieve_and_generate(self, question: str):
        """Convenience method to retrieve and generate in one call."""
        logger.info(f"Processing complete question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Create initial state
            state = {"question": question, "context": [], "answer": ""}
            
            # Retrieve documents
            retrieve_result = self.retrieve(state)
            state.update(retrieve_result)
            
            # Generate answer
            generate_result = self.generate(state)
            state.update(generate_result)
            
            logger.info("Question processed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {e}")
            raise
    
    def get_stats(self):
        """Get statistics about the vector store."""
        try:
            if hasattr(self.vector_store, 'count'):
                doc_count = self.vector_store.count()
            else:
                doc_count = "Unknown"
            
            stats = {
                "vector_store_documents": doc_count,
                "llm_model": getattr(self.llm, 'model_name', 'Unknown'),
                "prompt_type": type(self.prompt).__name__
            }
            
            logger.info(f"LLMBuilder stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}