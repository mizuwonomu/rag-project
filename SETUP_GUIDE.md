# RAG Project - Setup and Usage Guide

## Prerequisites

Before running this RAG system, you need:

1. **Python 3.8+** (tested with Python 3.12.3)
2. **Google API Key** for Gemini AI services
3. **Documents to query** (text/markdown files)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mizuwonomu/rag-project.git
cd rag-project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

**To get a Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 5. Prepare Your Documents

Create a `data/` directory and add your text files:
```bash
mkdir data
# Copy your .txt or .md files to the data/ directory
```

### 6. Initialize the Vector Database
```bash
python src/ingest.py
```

This will:
- Process all documents in the `data/` directory
- Create vector embeddings using Google Gemini
- Store embeddings in Chroma DB (`chroma_db/` directory)

## Usage

### Option 1: Command Line Interface
```bash
python main.py
```

This provides a simple terminal interface where you can:
- Type questions about your documents
- Receive AI-generated answers
- Type 'exit' to quit

### Option 2: Web Interface (Recommended)
```bash
streamlit run app.py
```

The web interface offers:
- Interactive question input
- Configurable parameters (k-value, temperature)
- Source document display
- Vietnamese language interface

## Configuration Options

### In the Web Interface (app.py):
- **k (Số lượng chunk tìm kiếm)**: Number of document chunks to retrieve (1-10)
- **Temperature**: Response creativity level (0.0-1.0)
  - 0.0 = Most factual, deterministic
  - 1.0 = Most creative, varied

### Document Processing (src/data_processing.py):
- **chunk_size**: Size of text chunks (default: 500 characters)
- **chunk_overlap**: Overlap between chunks (default: 50 characters)

## Project Structure

```
rag-project/
├── src/                    # Core source code
│   ├── data_processing.py  # Document cleaning and chunking
│   ├── vector_store.py     # Chroma DB operations
│   ├── qa_chain.py         # RAG pipeline implementation
│   └── ingest.py          # Document ingestion script
├── data/                   # Your document files (.txt, .md)
├── chroma_db/             # Vector database (auto-generated)
├── main.py                # Command-line interface
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
└── README.md             # Basic project info
```

## Adding New Documents

To add new documents to your system:

1. Place new `.txt` files in the `data/` directory
2. Run the ingestion script:
   ```bash
   python src/ingest.py
   ```

The system will:
- Detect new files automatically
- Process only new documents (incremental updates)
- Update the vector database

## Troubleshooting

### Common Issues:

1. **"DefaultCredentialsError"**
   - Solution: Ensure your `GOOGLE_API_KEY` is set in the `.env` file

2. **"No module named '...'"**
   - Solution: Install requirements: `pip install -r requirements.txt`

3. **"Lỗi: Thư mục data không tồn tại"**
   - Solution: Create the data directory: `mkdir data`

4. **Empty responses or poor quality answers**
   - Check if documents were processed correctly
   - Adjust k-value (try higher values like 5-7)
   - Verify document content is relevant to your questions

### Checking System Status:

```bash
# Verify dependencies
python -c "from src.data_processing import doc_cleaning; print('✓ Dependencies OK')"

# Check processed documents
ls -la chroma_db/  # Should contain database files

# Verify document count
python -c "
import glob
print(f'Documents found: {len(glob.glob(\"data/*.txt\"))}')
"
```

## Best Practices

1. **Document Quality**: Use well-structured, informative documents
2. **File Naming**: Use descriptive filenames for your documents
3. **Question Phrasing**: Ask specific, clear questions for better results
4. **Parameter Tuning**: Experiment with k-values and temperature settings
5. **Regular Updates**: Re-run ingestion when adding new documents

## Language Support

The system supports:
- **Interface**: Vietnamese (can be easily changed)
- **Documents**: Any language supported by Google Gemini
- **Questions**: Vietnamese, English, and other major languages

## Performance Notes

- **First Run**: Initial setup and embedding creation takes time
- **Subsequent Queries**: Fast response times after setup
- **Document Size**: Large documents are automatically chunked for optimal performance
- **Concurrent Users**: Streamlit app supports multiple simultaneous users

## Security Notes

- Keep your `.env` file secure and never commit it to version control
- The `chroma_db/` directory contains your processed documents
- All processing happens locally except for Google AI API calls