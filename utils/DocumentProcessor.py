import bs4
import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Get logger for this module
logger = logging.getLogger(__name__)

def get_docs(web_paths="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    """Get the documents loaded from the web."""
    logger.info(f"Loading documents from: {web_paths}")
    
    try:
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=(web_paths,),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        
        if not docs:
            logger.error("No documents loaded from the web")
            raise ValueError("No documents were loaded from the specified URL")
        
        logger.info(f"Successfully loaded {len(docs)} document(s)")
        logger.info(f"Total characters in first document: {len(docs[0].page_content)}")
        
        return docs
        
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

def text_split(docs):
    """Split the documents into smaller chunks."""
    logger.info(f"Splitting {len(docs)} documents into chunks")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        
        logger.info(f"Split documents into {len(all_splits)} chunks")
        logger.debug(f"Average chunk size: {sum(len(split.page_content) for split in all_splits) / len(all_splits):.0f} characters")
        
        return all_splits
        
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise

def vectorize_docs(docs, vector_store):
    """Vectorize the documents and add them to the vector store."""
    logger.info(f"Vectorizing {len(docs)} document chunks")
    
    try:
        # Extract texts and metadatas separately
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        # Add documents to vector store in batch
        document_ids = vector_store.add_texts(texts, metadatas)
        
        # Check if vector store has a count method, otherwise get length of returned IDs
        try:
            total_docs = vector_store.count() if hasattr(vector_store, 'count') else len(document_ids)
            logger.info(f"Successfully vectorized documents. Total documents in vector store: {total_docs}")
        except Exception as count_error:
            logger.warning(f"Could not get vector store count: {count_error}")
            logger.info(f"Successfully vectorized {len(document_ids)} documents")
        
        return document_ids
        
    except Exception as e:
        logger.error(f"Error vectorizing documents: {e}")
        raise

# Alternative function for multiple URLs
def get_docs_multiple(web_paths_list):
    """Get documents from multiple web URLs."""
    logger.info(f"Loading documents from {len(web_paths_list)} URLs")
    
    all_docs = []
    for i, web_path in enumerate(web_paths_list):
        try:
            logger.info(f"Loading document {i+1}/{len(web_paths_list)}: {web_path}")
            docs = get_docs(web_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Failed to load document from {web_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(all_docs)} documents from {len(web_paths_list)} URLs")
    return all_docs

# Utility function to validate document content
def validate_docs(docs):
    """Validate that documents have content."""
    logger.info(f"Validating {len(docs)} documents")
    
    valid_docs = []
    for i, doc in enumerate(docs):
        if not doc.page_content or len(doc.page_content.strip()) == 0:
            logger.warning(f"Document {i} is empty or contains only whitespace")
            continue
        
        if len(doc.page_content) < 50:  # Minimum content threshold
            logger.warning(f"Document {i} has very little content ({len(doc.page_content)} characters)")
        
        valid_docs.append(doc)
    
    logger.info(f"Validated documents: {len(valid_docs)} valid out of {len(docs)} total")
    return valid_docs