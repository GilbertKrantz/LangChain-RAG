import bs4
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredCSVLoader,  # Added for CSV support
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Get logger for this module
logger = logging.getLogger(__name__)


def is_url(path):
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_directory(path):
    """Check if the given path is a directory."""
    return os.path.isdir(path)


def get_docs(paths="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    """Get the documents loaded from web URLs or local file paths."""
    if isinstance(paths, str):
        paths = [paths]

    logger.info(f"Loading documents from: {paths}")

    all_docs = []

    for path in paths:
        try:
            if is_url(path):
                docs = _load_web_docs(path)
            elif is_directory(path):
                docs = _load_directory_docs(path)
            else:
                docs = _load_file_docs(path)

            all_docs.extend(docs)

        except Exception as e:
            logger.error(f"Error loading documents from {path}: {e}")
            raise

    if not all_docs:
        logger.error("No documents loaded from any source")
        raise ValueError("No documents were loaded from the specified paths")

    logger.info(f"Successfully loaded {len(all_docs)} document(s) total")
    return all_docs


def _load_web_docs(web_path):
    """Load documents from a web URL."""
    logger.info(f"Loading web document from: {web_path}")

    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(web_path,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No documents were loaded from URL: {web_path}")

    # Add source metadata
    for doc in docs:
        doc.metadata["source_type"] = "web"
        doc.metadata["original_source"] = web_path

    logger.info(f"Loaded {len(docs)} web document(s)")
    return docs


def _load_file_docs(file_path):
    """Load documents from a local file."""
    logger.info(f"Loading local file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file extension to determine loader type
    file_ext = Path(file_path).suffix.lower()

    if file_ext in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml"]:
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".csv":
        try:
            # Use pandas to read CSV and convert to text
            df = pd.read_csv(file_path)
            # Convert DataFrame to string representation
            content = df.to_string(index=False)
            # Create a Document object manually
            from langchain_core.documents import Document

            doc = Document(
                page_content=content,
                metadata={
                    "source_type": "file",
                    "original_source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_extension": file_ext,
                },
            )
            docs = [doc]
        except Exception as e:
            logger.error(f"Failed to load CSV with pandas: {e}")
            raise ValueError(f"Unable to load CSV file: {file_path}")
    else:
        # Try TextLoader as fallback for other text-based files
        try:
            loader = TextLoader(file_path, encoding="utf-8")
        except Exception:
            raise ValueError(f"Unsupported file type: {file_ext}")

    if file_ext != ".csv":  # For non-CSV files, use the loader
        docs = loader.load()

    if not docs:
        raise ValueError(f"No documents were loaded from file: {file_path}")

    # Add source metadata for non-CSV files
    if file_ext != ".csv":
        for doc in docs:
            doc.metadata["source_type"] = "file"
            doc.metadata["original_source"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_extension"] = file_ext

    logger.info(f"Loaded {len(docs)} document(s) from file")
    return docs


def _load_directory_docs(directory_path):
    """Load documents from all files in a directory."""
    logger.info(f"Loading documents from directory: {directory_path}")

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Define supported file extensions
    supported_extensions = [
        ".txt",
        ".md",
        ".py",
        ".js",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".pdf",
        ".csv",  # Added CSV support
    ]

    # Load text-based files
    text_loader = DirectoryLoader(
        directory_path,
        glob="**/*.[tT][xX][tT]|**/*.[mM][dD]|**/*.[pP][yY]|**/*.[jJ][sS]|**/*.[hH][tT][mM][lL]|**/*.[cC][sS][sS]|**/*.[jJ][sS][oO][nN]|**/*.[xX][mM][lL]",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )

    # Load PDFs separately
    pdf_docs = []
    try:
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )
        pdf_docs = pdf_loader.load()
    except Exception as e:
        logger.warning(f"Failed to load PDFs with DirectoryLoader: {e}")

    # Load CSVs separately
    csv_docs = []
    try:
        csv_loader = DirectoryLoader(
            directory_path,
            glob="**/*.csv",
            loader_cls=UnstructuredCSVLoader,
            show_progress=True,
            use_multithreading=True,
        )
        csv_docs = csv_loader.load()
    except Exception as e:
        logger.warning(f"Failed to load CSVs with DirectoryLoader: {e}")

    # Load text-based files
    try:
        docs = text_loader.load()
    except Exception as e:
        logger.warning(f"Text DirectoryLoader failed, trying manual loading: {e}")
        docs = _load_directory_manually(directory_path, supported_extensions)

    # Combine all documents
    docs.extend(pdf_docs)
    docs.extend(csv_docs)

    if not docs:
        raise ValueError(f"No documents were loaded from directory: {directory_path}")

    # Add source metadata
    for doc in docs:
        if "source_type" not in doc.metadata:
            doc.metadata["source_type"] = "directory"
        if "original_source" not in doc.metadata:
            doc.metadata["original_source"] = directory_path

    logger.info(f"Loaded {len(docs)} document(s) from directory")
    return docs


def _load_directory_manually(directory_path, supported_extensions):
    """Manually load files from directory if DirectoryLoader fails."""
    docs = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file_path).suffix.lower()

            if file_ext in supported_extensions:
                try:
                    file_docs = _load_file_docs(file_path)
                    docs.extend(file_docs)
                except Exception as e:
                    logger.warning(f"Failed to load file {file_path}: {e}")
                    continue

    return docs


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
        logger.debug(
            f"Average chunk size: {sum(len(split.page_content) for split in all_splits) / len(all_splits):.0f} characters"
        )

        total_documents = len(all_splits)
        third = total_documents // 3

        for i, document in enumerate(all_splits):
            if i < third:
                document.metadata["section"] = "beginning"
            elif i < 2 * third:
                document.metadata["section"] = "middle"
            else:
                document.metadata["section"] = "end"

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
            total_docs = (
                vector_store.count()
                if hasattr(vector_store, "count")
                else len(document_ids)
            )
            logger.info(
                f"Successfully vectorized documents. Total documents in vector store: {total_docs}"
            )
        except Exception as count_error:
            logger.warning(f"Could not get vector store count: {count_error}")
            logger.info(f"Successfully vectorized {len(document_ids)} documents")

        return document_ids

    except Exception as e:
        logger.error(f"Error vectorizing documents: {e}")
        raise


def get_docs_multiple(paths_list):
    """Get documents from multiple web URLs and/or local file paths."""
    logger.info(f"Loading documents from {len(paths_list)} sources")

    all_docs = []
    failed_sources = []

    for i, path in enumerate(paths_list):
        try:
            logger.info(f"Loading source {i+1}/{len(paths_list)}: {path}")
            docs = get_docs(path)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Failed to load documents from {path}: {e}")
            failed_sources.append(path)
            continue

    if failed_sources:
        logger.warning(
            f"Failed to load from {len(failed_sources)} sources: {failed_sources}"
        )

    logger.info(
        f"Successfully loaded {len(all_docs)} documents from {len(paths_list) - len(failed_sources)}/{len(paths_list)} sources"
    )
    return all_docs


def validate_docs(docs):
    """Validate that documents have content."""
    logger.info(f"Validating {len(docs)} documents")

    valid_docs = []
    for i, doc in enumerate(docs):
        if not doc.page_content or len(doc.page_content.strip()) == 0:
            logger.warning(f"Document {i} is empty or contains only whitespace")
            continue

        if len(doc.page_content) < 50:  # Minimum content threshold
            logger.warning(
                f"Document {i} has very little content ({len(doc.page_content)} characters)"
            )

        valid_docs.append(doc)

    logger.info(
        f"Validated documents: {len(valid_docs)} valid out of {len(docs)} total"
    )
    return valid_docs


def get_source_info(docs):
    """Get information about document sources."""
    source_info = {"web": 0, "file": 0, "directory": 0, "total": len(docs)}

    file_types = {}

    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        if source_type in source_info:
            source_info[source_type] += 1

        file_ext = doc.metadata.get("file_extension", "")
        if file_ext:
            file_types[file_ext] = file_types.get(file_ext, 0) + 1

    source_info["file_types"] = file_types

    logger.info(f"Source distribution: {source_info}")
    return source_info
