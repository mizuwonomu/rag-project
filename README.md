# RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using Chroma DB and Google's Gemini model.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mizuwonomu/rag-project.git
   cd rag-project
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

5. **Set up the Chroma DB**
   ```bash
   python setup.py
   ```
   This will process your documents and create the vector store.

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

- `main.py`: Main application script
- `setup.py`: Script to set up the Chroma DB
- `src/`: Source code directory
  - `data_processing.py`: Document processing logic
  - `vector_store.py`: Chroma DB setup and management
  - `qa_chain.py`: Question-answering chain implementation
- `chroma_db/`: Directory containing the vector store (not versioned)
- `.env`: Environment variables (not versioned)

## Notes

- The `chroma_db` directory is excluded from version control. You'll need to regenerate it using `setup.py` when cloning the repository.
- Make sure to keep your `.env` file secure and never commit it to version control.
