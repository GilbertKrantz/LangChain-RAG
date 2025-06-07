import logging
import json
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from pydantic import BaseModel, Field
from utils.SearchQuery import Search

logger = logging.getLogger(__name__)


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str
    reasoning: str
    stream: bool


class SearchQuery(BaseModel):
    query: str = Field(description="The search query string")
    section: str = Field(description="The section to search in", default="")
    is_conversational: bool = Field(
        description="Whether the query is a conversational greeting", default=False
    )


class LLMBuilder:
    def __init__(self, vector_store, prompt, llm):
        self.vector_store = vector_store
        self.prompt = prompt
        self.llm = llm
        logger.info("LLMBuilder initialized successfully")

    @classmethod
    def from_config(cls, config):
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
        logger.info(
            f"Analyzing question with parser: {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )
        try:
            parser = PydanticOutputParser(pydantic_object=SearchQuery)
            query_prompt = PromptTemplate(
                template="""
                Analyze the following question and extract:
                1. A search query string that would help find relevant information (use the original question for conversational greetings)
                2. A section name if the question refers to a specific section (leave empty if not specified)
                3. A boolean indicating if the question is a conversational greeting (e.g., "hello", "how are you")
                
                Question: {question}
                
                {format_instructions}
                """,
                input_variables=["question"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )
            chain = query_prompt | self.llm | parser
            query_result = chain.invoke({"question": state["question"]})
            query_dict = {
                "query": query_result.query,
                "section": query_result.section,
                "is_conversational": query_result.is_conversational,
            }
            logger.info(f"Query analysis result: {query_dict}")
            return {"query": query_dict}
        except Exception as e:
            logger.error(f"Error in analyze_query_with_parser: {e}")
            return self.analyze_query_simple(state)

    def analyze_query_simple(self, state: State):
        logger.info(
            f"Analyzing question (simple): {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )
        try:
            analysis_prompt = f"""
            Based on this question, provide a search query that would help find relevant information.
            If the question is a conversational greeting (e.g., "hello", "how are you"), use the original question as the query.
            Question: {state['question']}
            
            Respond with just the search query, nothing else.
            """
            response = self.llm.invoke(analysis_prompt)
            if hasattr(response, "content"):
                query_text = response.content.strip()
            elif isinstance(response, str):
                query_text = response.strip()
            else:
                query_text = str(response).strip()
            is_conversational = any(
                greeting in state["question"].lower()
                for greeting in ["hello", "how are you", "hi"]
            )
            query_dict = {
                "query": query_text,
                "section": "",
                "is_conversational": is_conversational,
            }
            logger.info(f"Generated simple query: {query_dict}")
            return {"query": query_dict}
        except Exception as e:
            logger.error(f"Error in analyze_query_simple: {e}")
            query_dict = {
                "query": state["question"],
                "section": "",
                "is_conversational": False,
            }
            return {"query": query_dict}

    def analyze_query_json(self, state: State):
        logger.info(
            f"Analyzing question with JSON: {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )
        try:
            parser = JsonOutputParser()
            json_prompt = PromptTemplate(
                template="""
                Analyze the following question and respond with a JSON object containing:
                - "query": a search query string that would help find relevant information (use original question for conversational greetings)
                - "section": a section name if mentioned in the question (empty string if not)
                - "is_conversational": a boolean indicating if the question is a conversational greeting (e.g., "hello", "how are you")
                
                Question: {question}
                
                {format_instructions}
                
                Respond only with valid JSON.
                """,
                input_variables=["question"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )
            chain = json_prompt | self.llm | parser
            query_result = chain.invoke({"question": state["question"]})
            query_dict = {
                "query": query_result.get("query", state["question"]),
                "section": query_result.get("section", ""),
                "is_conversational": query_result.get("is_conversational", False),
            }
            logger.info(f"JSON query analysis result: {query_dict}")
            return {"query": query_dict}
        except Exception as e:
            logger.error(f"Error in analyze_query_json: {e}")
            return self.analyze_query_simple(state)

    def analyze_query(self, state: State):
        logger.info(
            f"Analyzing question: {state['question'][:100]}{'...' if len(state['question']) > 100 else ''}"
        )
        if hasattr(self.llm, "with_structured_output"):
            try:
                structured_llm = self.llm.with_structured_output(Search)
                query = structured_llm.invoke(state["question"])
                query_dict = {
                    "query": query.query,
                    "section": query.section,
                    "is_conversational": False,
                }
                logger.info(f"Structured query analysis result: {query_dict}")
                return {"query": query_dict}
            except NotImplementedError:
                logger.info(
                    "LLM doesn't support with_structured_output, using alternative approach"
                )
            except Exception as e:
                logger.warning(
                    f"Structured output failed: {e}, using alternative approach"
                )
        try:
            return self.analyze_query_with_parser(state)
        except Exception as e:
            logger.warning(f"Parser approach failed: {e}, trying JSON approach")
        try:
            return self.analyze_query_json(state)
        except Exception as e:
            logger.warning(f"JSON approach failed: {e}, using simple approach")
        return self.analyze_query_simple(state)

    def retrieve(self, state: State):
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
            if isinstance(query, dict):
                search_query = query.get("query", state["question"])
                section_filter = query.get("section", "")
                is_conversational = query.get("is_conversational", False)
            else:
                search_query = getattr(query, "query", state["question"])
                section_filter = getattr(query, "section", "")
                is_conversational = getattr(query, "is_conversational", False)
            if is_conversational:
                logger.info(
                    "Conversational query detected, skipping document retrieval"
                )
                return {"context": []}
            if section_filter:
                retrieved_docs = self.vector_store.similarity_search(
                    search_query,
                    filter=lambda doc: doc.metadata.get("section") == section_filter,
                )
            else:
                retrieved_docs = self.vector_store.similarity_search(search_query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            logger.debug(
                f"Document lengths: {[len(doc.page_content) for doc in retrieved_docs]}"
            )
            return {"context": retrieved_docs}
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return {"context": []}

    def generate(self, state: State):
        question = state["question"]
        context_docs = state.get("context", [])
        stream = state.get("stream", False)
        logger.info(
            f"Generating answer for question with {len(context_docs)} context documents, stream={stream}"
        )
        try:
            if context_docs:
                docs_content = "\n\n".join(doc.page_content for doc in context_docs)
                logger.debug(f"Context length: {len(docs_content)} characters")
            else:
                docs_content = "No relevant context found."
                logger.warning(
                    "No context documents available for generation, Use Converational Context"
                )
                docs_content = "You are a helpful assistant. Please answer the question based on your knowledge."

            # Prepare prompt input
            prompt_input = {"question": question, "context": docs_content}
            messages = self.prompt.invoke(prompt_input)

            # Convert ChatPromptValue to a compatible format
            if hasattr(messages, "messages"):
                # Extract the content of the last message (assuming it's the user prompt)
                prompt_text = messages.messages[-1].content
            else:
                prompt_text = str(messages)

            logger.debug(f"Prompt prepared, invoking LLM")
            if stream and hasattr(self.llm, "_stream"):
                answer = ""
                reasoning = ""
                for chunk in self.llm._stream(prompt_text):
                    answer += chunk.text
                    if chunk.generation_info and "reasoning" in chunk.generation_info:
                        reasoning += chunk.generation_info["reasoning"]
                    logger.debug(f"Streamed chunk received (length: {len(chunk.text)})")
                result = {"answer": answer.strip(), "reasoning": reasoning.strip()}
            else:
                response = self.llm.invoke(prompt_text)
                if hasattr(response, "content"):
                    answer = response.content
                    reasoning = (
                        response.generation_info.get("reasoning", "")
                        if hasattr(response, "generation_info")
                        else ""
                    )
                elif isinstance(response, str):
                    answer = response
                    reasoning = ""
                else:
                    answer = str(response)
                    reasoning = ""
                result = {"answer": answer.strip(), "reasoning": reasoning.strip()}
            logger.info(
                f"Answer generated successfully (length: {len(result['answer'])} characters)"
            )
            return result
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while generating an answer: {str(e)}",
                "reasoning": "",
            }

    def retrieve_and_generate(self, question: str):
        logger.info(
            f"Processing complete question: {question[:100]}{'...' if len(question) > 100 else ''}"
        )
        try:
            state = {
                "question": question,
                "context": [],
                "answer": "",
                "reasoning": "",
                "stream": False,
            }
            retrieve_result = self.retrieve(state)
            state.update(retrieve_result)
            generate_result = self.generate(state)
            state.update(generate_result)
            logger.info("Question processed successfully")
            return state
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {e}")
            raise

    def get_stats(self):
        try:
            if hasattr(self.vector_store, "count"):
                doc_count = self.vector_store.count()
            else:
                doc_count = "Unknown"
            stats = {
                "vector_store_documents": doc_count,
                "llmlm_model": getattr(self.llm, "model_name", "Unknown"),
                "prompt_type": type(self.prompt).__name__,
            }
            logger.info(f"LLMBuilder stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
