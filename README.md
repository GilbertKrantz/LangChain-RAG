# RAG LangChain Application

A powerful Retrieval-Augmented Generation (RAG) application built with LangChain, Google Gemini, and LangGraph that can process documents from multiple sources and answer questions based on their content.

## 🚀 Features

- **Multi-Source Document Loading**: Support for web URLs, local files, and directories
- **Google Gemini Integration**: Uses Google's Gemini 1.5 Flash model for LLM and embeddings
- **Smart Document Processing**: Automatic text splitting and vectorization
- **Interactive Chat Interface**: Command-line interface for asking questions
- **Comprehensive Logging**: Detailed logging with timestamps and multiple levels
- **Error Handling**: Robust error handling with graceful fallbacks
- **Flexible Configuration**: Environment variable-based configuration
- **Multiple File Format Support**: .txt, .md, .py, .js, .html, .css, .json, .xml

## 📋 Prerequisites

- Python 3.12.9 or higher
- UV package manager (recommended) or pip
- Google AI API key

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag-langchain
   ```

2. **Install dependencies using UV:**
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   export GOOGLE_API_KEY="your-google-api-key-here"
   ```

## 🔑 Getting Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key and set it as an environment variable

## 🚀 Usage

### Basic Usage

Run the application with default settings (loads from a default web URL):

```bash
uv run main.py
```

### Custom Document Sources

You can specify custom document sources using environment variables:

#### Web URLs
```bash
export DOCUMENT_SOURCES="https://example.com/page1,https://example.com/page2"
uv run main.py
```

#### Local Files and Directories
```bash
export DOCUMENT_PATHS="/path/to/file.txt,/path/to/directory"
uv run main.py
```

#### Mixed Sources
```bash
export DOCUMENT_SOURCES="https://example.com/page"
export DOCUMENT_PATHS="/path/to/local/docs"
uv run main.py
```

### Quiet Mode

For minimal output (errors only):

```bash
uv run main.py --quiet
# or
uv run main.py -q
# or
export QUIET_MODE=true && uv run main.py
```

### Custom Log Level

```bash
export LOGGING_LEVEL=DEBUG
uv run main.py
```

## 💬 Interactive Commands

Once the application is running, you can use these commands:

- Type your question and press Enter to get an answer
- `help` or `h` - Show usage information
- `exit`, `quit`, or `q` - Exit the application
- `Ctrl+C` - Force exit

## 📁 Project Structure

```
rag-langchain/
├── main.py                    # Main application entry point
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # This file
├── logs/                     # Generated log files (created automatically)
└── utils/                    # Utility modules
    ├── DocumentProcessor.py  # Document loading and processing
    ├── LLMBuilder.py        # LLM and RAG pipeline builder
    └── SearchQuery.py       # Search query type definitions
```

## 🛠️ Architecture

The application uses a graph-based architecture with LangGraph:

1. **Query Analysis**: Analyzes the user's question to extract search parameters
2. **Document Retrieval**: Searches the vector store for relevant documents
3. **Answer Generation**: Uses retrieved context to generate comprehensive answers

### Key Components

- **DocumentProcessor**: Handles loading documents from various sources
- **LLMBuilder**: Manages the RAG pipeline and LLM interactions
- **Vector Store**: In-memory vector storage with Google Gemini embeddings
- **LangGraph**: Orchestrates the multi-step RAG process

## 📊 Supported File Types

The application can process the following file types:
- Text files (`.txt`)
- Markdown files (`.md`)
- Python files (`.py`)
- JavaScript files (`.js`)
- HTML files (`.html`)
- CSS files (`.css`)
- JSON files (`.json`)
- XML files (`.xml`)

## 🔍 How It Works

1. **Document Loading**: Documents are loaded from specified sources (web URLs, files, or directories)
2. **Text Splitting**: Large documents are split into smaller chunks for better processing
3. **Vectorization**: Text chunks are converted to embeddings using Google Gemini
4. **Query Processing**: User questions are analyzed to extract search parameters
5. **Retrieval**: Relevant document chunks are retrieved based on similarity search
6. **Generation**: Final answers are generated using the retrieved context

## 📝 Logging

The application generates detailed logs in the `logs/` directory with timestamps. Log levels can be controlled via the `LOGGING_LEVEL` environment variable:

- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages only
- `CRITICAL`: Critical errors only

## 🐛 Troubleshooting

### Common Issues

1. **"Environment variable 'GOOGLE_API_KEY' is not set"**
   - Ensure you've set your Google API key as an environment variable

2. **"No documents could be loaded from any source"**
   - Check that your file paths exist and are accessible
   - Verify web URLs are reachable
   - Ensure files are in supported formats

3. **"LLM connection test failed"**
   - Verify your Google API key is valid
   - Check your internet connection
   - Ensure the API key has appropriate permissions

4. **Memory issues with large documents**
   - Try processing smaller document sets
   - Increase chunk size in the text splitter
   - Use quiet mode to reduce memory usage

### Getting Help

- Check the logs in the `logs/` directory for detailed error information
- Use `--quiet` mode to see only critical errors
- Ensure all environment variables are properly set

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Google Gemini](https://ai.google.dev/)
- Uses [LangSmith Hub](https://smith.langchain.com/hub) for RAG prompts

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Review the logs for detailed error information
3. Open an issue on the repository with relevant log outputs

---

**Happy querying! 🎉**