import logging
import json
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from pydantic import BaseModel, Field

from utils.SearchQuery import Search

# Get logger for this module
logger = logging.getLogger(__name__)


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


class SearchQuery(BaseModel):
    """Pydantic model for search query structure"""

    query: str = Field(description="The search query string")
    section: str = Field(description="The section to search in", default="")


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

    def analyze_query_with_parser(self, state: State):
        """Analyze the question using output parser (Method 1)."""
        logger.info(
            f"Analyzing question with parser: {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )

        try:
            # Create parser for SearchQuery
            parser = PydanticOutputParser(pydantic_object=SearchQuery)

            # Create prompt template with format instructions
            query_prompt = PromptTemplate(
                template="""
                Analyze the following question and extract:
                1. A search query string that would help find relevant information
                2. A section name if the question refers to a specific section (leave empty if not specified)
                
                Question: {question}
                
                {format_instructions}
                """,
                input_variables=["question"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            # Create chain
            chain = query_prompt | self.llm | parser

            # Invoke chain
            query_result = chain.invoke({"question": state["question"]})

            # Convert to dict format expected by the rest of the code
            query_dict = {"query": query_result.query, "section": query_result.section}

            return {"query": query_dict}

        except Exception as e:
            logger.error(f"Error in analyze_query_with_parser: {e}")
            # Fallback to simple approach
            return self.analyze_query_simple(state)

    def analyze_query_simple(self, state: State):
        """Simple fallback method without structured output (Method 2)."""
        logger.info(
            f"Analyzing question (simple): {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )

        try:
            # Create a simple prompt to extract query information
            analysis_prompt = f"""
            Based on this question, provide a search query that would help find relevant information:
            Question: {state['question']}
            
            Respond with just the search query, nothing else.
            """

            # Get response from LLM
            response = self.llm.invoke(analysis_prompt)

            # Handle different response types
            if hasattr(response, "content"):
                query_text = response.content.strip()
            elif isinstance(response, str):
                query_text = response.strip()
            else:
                query_text = str(response).strip()

            # Create query dict
            query_dict = {"query": query_text, "section": ""}  # Default empty section

            logger.info(f"Generated simple query: {query_dict['query']}")
            return {"query": query_dict}

        except Exception as e:
            logger.error(f"Error in analyze_query_simple: {e}")
            # Ultimate fallback - use the original question
            query_dict = {"query": state["question"], "section": ""}
            return {"query": query_dict}

    def analyze_query_json(self, state: State):
        """Analyze query using JSON output parsing (Method 3)."""
        logger.info(
            f"Analyzing question with JSON: {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )

        try:
            # Create JSON output parser
            parser = JsonOutputParser()

            # Create prompt that asks for JSON response
            json_prompt = PromptTemplate(
                template="""
                Analyze the following question and respond with a JSON object containing:
                - "query": a search query string that would help find relevant information
                - "section": a section name if mentioned in the question (empty string if not)
                
                Question: {question}
                
                {format_instructions}
                
                Respond only with valid JSON.
                """,
                input_variables=["question"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            # Create chain
            chain = json_prompt | self.llm | parser

            # Invoke chain
            query_result = chain.invoke({"question": state["question"]})

            # Ensure we have the required keys
            query_dict = {
                "query": query_result.get("query", state["question"]),
                "section": query_result.get("section", ""),
            }

            return {"query": query_dict}

        except Exception as e:
            logger.error(f"Error in analyze_query_json: {e}")
            # Fallback to simple approach
            return self.analyze_query_simple(state)

    def analyze_query(self, state: State):
        """Main analyze_query method that tries different approaches."""
        logger.info(
            f"Analyzing question: {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )

        # Check if LLM supports structured output
        if hasattr(self.llm, "with_structured_output"):
            try:
                # Try the original approach first
                structured_llm = self.llm.with_structured_output(Search)
                query = structured_llm.invoke(state["question"])
                return {"query": query}
            except NotImplementedError:
                logger.info(
                    "LLM doesn't support with_structured_output, using alternative approach"
                )
            except Exception as e:
                logger.warning(
                    f"Structured output failed: {e}, using alternative approach"
                )

        # Try parser-based approach
        try:
            return self.analyze_query_with_parser(state)
        except Exception as e:
            logger.warning(f"Parser approach failed: {e}, trying JSON approach")

        # Try JSON approach
        try:
            return self.analyze_query_json(state)
        except Exception as e:
            logger.warning(f"JSON approach failed: {e}, using simple approach")

        # Final fallback
        return self.analyze_query_simple(state)

    def retrieve(self, state: State):
        """Retrieve relevant documents based on the question."""
        logger.info("Starting document retrieval process")
        query = state.get("query")
        if not query:
            logger.warning("No query found in state, analyzing question")
            query_result = self.analyze_query(state)
            query = query_result["query"]

        logger.info(
            f"Retrieving documents for query: {query.get('query', '')[:100]}{'...' if len(query.get('query', '')) > 100 else ''}"
        )

        try:
            # Handle both dict and object query formats
            if isinstance(query, dict):
                search_query = query.get("query", state["question"])
                section_filter = query.get("section", "")
            else:
                search_query = getattr(query, "query", state["question"])
                section_filter = getattr(query, "section", "")

            # Perform similarity search
            if section_filter:
                retrieved_docs = self.vector_store.similarity_search(
                    search_query,
                    filter=lambda doc: doc.metadata.get("section") == section_filter,
                )
            else:
                # If no section filter, search without filtering
                retrieved_docs = self.vector_store.similarity_search(search_query)

            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            logger.debug(
                f"Document lengths: {[len(doc.page_content) for doc in retrieved_docs]}"
            )

            return {"context": retrieved_docs}

        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            # Return empty context to allow the chain to continue
            return {"context": []}

    def generate(self, state: State):
        """Generate answer based on retrieved context and question."""
        question = state["question"]
        context_docs = state.get("context", [])

        logger.info(
            f"Generating answer for question with {len(context_docs)} context documents"
        )

        try:
            # Prepare context from retrieved documents
            if context_docs:
                docs_content = "\n\n".join(doc.page_content for doc in context_docs)
                logger.debug(f"Context length: {len(docs_content)} characters")
            else:
                docs_content = "No relevant context found."
                logger.warning("No context documents available for generation")

            # Invoke the prompt with question and context
            messages = self.prompt.invoke(
                {"question": question, "context": docs_content}
            )

            logger.debug(f"Prompt prepared, invoking LLM")

            # Invoke the LLM
            response = self.llm.invoke(messages)

            # Handle different response types
            if hasattr(response, "content"):
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
                logger.warning(
                    f"Unexpected response type: {type(response)}, converted to string"
                )

            logger.info(
                f"Answer generated successfully (length: {len(answer)} characters)"
            )
            return {"answer": answer}

        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            # Return a fallback answer
            fallback_answer = f"I apologize, but I encountered an error while generating an answer: {str(e)}"
            return {"answer": fallback_answer}

    def retrieve_and_generate(self, question: str):
        """Convenience method to retrieve and generate in one call."""
        logger.info(
            f"Processing complete question: {question[:100]}{'...' if len(question) > 100 else ''}"
        )

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
            if hasattr(self.vector_store, "count"):
                doc_count = self.vector_store.count()
            else:
                doc_count = "Unknown"

            stats = {
                "vector_store_documents": doc_count,
                "llm_model": getattr(self.llm, "model_name", "Unknown"),
                "prompt_type": type(self.prompt).__name__,
            }

            logger.info(f"LLMBuilder stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
