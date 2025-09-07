# RAG Project - Comprehensive Repository Explanation

## Overview

This repository implements a **Retrieval-Augmented Generation (RAG) system** that combines document retrieval with AI-powered question answering. The system allows users to ask questions about a collection of documents and receive intelligent, contextual answers backed by relevant source material.

## 🏗️ Architecture

The RAG system follows a classic retrieval-augmented generation pipeline:

```
Documents → Processing → Vector Storage → Retrieval → Answer Generation
```

1. **Document Ingestion**: Text documents are cleaned and split into chunks
2. **Embedding Creation**: Document chunks are converted to vector embeddings using Google's Gemini
3. **Vector Storage**: Embeddings are stored in Chroma DB for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against stored documents
5. **Answer Generation**: Retrieved context is combined with the question to generate answers using Gemini AI

## 🛠️ Technology Stack

### Core Technologies
- **LangChain**: Framework for building language model applications
- **Chroma DB**: Vector database for storing and retrieving document embeddings
- **Google Gemini AI**: Both embedding model (`gemini-embedding-001`) and language model (`gemini-2.5-flash`)
- **Streamlit**: Web interface framework
- **Python**: Primary programming language

### Key Dependencies
- `langchain-chroma`: Chroma integration for LangChain
- `langchain-google-genai`: Google AI integration
- `langchain-text-splitters`: Text chunking capabilities
- `streamlit`: Web UI framework
- `beautifulsoup4`: HTML/Markdown cleaning
- `python-dotenv`: Environment variable management

## 📁 Project Structure

```
rag-project/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── data_processing.py       # Document cleaning and chunking
│   ├── vector_store.py          # Chroma DB operations
│   ├── qa_chain.py              # RAG chain implementation
│   └── ingest.py               # Document ingestion pipeline
├── main.py                      # Command-line interface
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
├── README.md                    # Basic setup instructions
├── .gitignore                   # Git ignore patterns
└── .env                         # Environment variables (not tracked)
```

### 📝 Component Details

#### `src/data_processing.py`
- **Purpose**: Document preprocessing and text chunking
- **Key Functions**:
  - `doc_cleaning()`: Converts markdown to clean text using BeautifulSoup
  - `chunking_doc()`: Splits documents into overlapping chunks with metadata
- **Features**: 
  - Markdown to HTML conversion
  - HTML tag removal
  - Configurable chunk size and overlap
  - Rich metadata tracking (source, timestamp, chunk position)

#### `src/vector_store.py`
- **Purpose**: Vector database operations
- **Key Functions**:
  - `store_embeddings()`: Creates and persists Chroma vector store
- **Features**:
  - Google Gemini embedding integration
  - Persistent storage to disk
  - Error handling and logging

#### `src/qa_chain.py`
- **Purpose**: Core RAG implementation
- **Key Functions**:
  - `get_chain()`: Creates the complete RAG pipeline
- **Features**:
  - Configurable retrieval parameters (k-value)
  - Temperature control for response generation
  - Parallel processing of context and questions
  - Vietnamese language support in prompts

#### `src/ingest.py`
- **Purpose**: Document ingestion and processing pipeline
- **Key Functions**:
  - `main()`: Processes new documents and updates vector store
  - `get_processed_files()`: Tracks already processed files
- **Features**:
  - Incremental processing (only new files)
  - Progress logging and tracking
  - Batch document processing

#### `main.py`
- **Purpose**: Command-line interface
- **Features**:
  - Interactive question-answering loop
  - Vietnamese language interface
  - Simple exit mechanism

#### `app.py`
- **Purpose**: Streamlit web interface
- **Features**:
  - Interactive web UI with Vietnamese language support
  - Configurable parameters (k-value, temperature)
  - Source document display
  - Real-time parameter adjustment
  - Expandable source citations

## 🚀 Key Features

### 1. **Intelligent Document Processing**
- Converts markdown documents to clean text
- Splits documents into semantically meaningful chunks
- Preserves context with configurable overlap between chunks
- Tracks comprehensive metadata for each chunk

### 2. **Advanced Retrieval**
- Uses Google Gemini embeddings for high-quality vector representations
- Similarity-based retrieval with configurable result count
- Persistent vector storage with Chroma DB
- Efficient incremental updates for new documents

### 3. **Contextual Answer Generation**
- Combines retrieved context with user questions
- Uses Google Gemini 2.5 Flash for natural language generation
- Configurable response temperature for creativity control
- Transparency with source document citations

### 4. **User-Friendly Interfaces**
- **Command Line**: Simple interactive terminal interface
- **Web UI**: Rich Streamlit interface with parameter controls
- **Bilingual Support**: Vietnamese interface with English capabilities

### 5. **Robust Configuration**
- Environment variable management with `.env` file
- Configurable chunking parameters
- Adjustable retrieval and generation settings
- Logging and error handling throughout

## 📊 Data Flow

### Document Ingestion Flow
```
Text Files → Markdown Processing → HTML Conversion → Text Extraction → 
Chunking → Embedding Generation → Vector Storage
```

### Query Processing Flow
```
User Question → Question Embedding → Vector Similarity Search → 
Context Retrieval → Prompt Construction → Answer Generation → Response Display
```

## 🔧 Configuration Options

### Chunking Parameters
- **chunk_size**: Size of each text chunk (default: 500 characters)
- **chunk_overlap**: Overlap between chunks (default: 50 characters)

### Retrieval Parameters
- **k**: Number of similar chunks to retrieve (configurable in UI: 1-10)
- **search_type**: Similarity search method

### Generation Parameters
- **temperature**: Creativity level for responses (configurable in UI: 0.0-1.0)
- **model**: Gemini model variant for generation

## 🌐 Language Support

The system is designed with Vietnamese language support:
- Interface text in Vietnamese
- Vietnamese prompt templates
- Multilingual document processing capabilities
- English and Vietnamese question answering

## 🔒 Security & Privacy

- Environment variables for API keys
- `.env` file excluded from version control
- Local vector storage (not cloud-dependent)
- No external data transmission except to Google AI APIs

## 📈 Scalability Considerations

- **Vector Storage**: Chroma DB provides efficient similarity search
- **Incremental Updates**: Only processes new documents
- **Modular Architecture**: Easy to swap components
- **Configurable Parameters**: Adaptable to different use cases

## 🎯 Use Cases

This RAG system is ideal for:
- **Document Q&A**: Querying large document collections
- **Knowledge Management**: Organizational knowledge bases
- **Research Assistance**: Academic paper analysis
- **Customer Support**: FAQ automation
- **Content Analysis**: Extracting insights from text collections

## 🔧 Extension Points

The modular architecture allows for easy extensions:
- **New Document Types**: Extend `data_processing.py` for PDFs, Word docs, etc.
- **Different LLMs**: Modify `qa_chain.py` to use other language models
- **Custom Embeddings**: Replace Google embeddings with alternatives
- **Enhanced UI**: Extend Streamlit interface with additional features
- **API Integration**: Add REST API endpoints for programmatic access

This RAG system provides a solid foundation for building intelligent document-based question-answering applications with modern AI technologies.