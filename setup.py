import os
import sys
from dotenv import load_dotenv
from src.vector_store import store_embeddings
from src.data_processing import process_documents

def setup_chroma_db():
    """
    Set up the Chroma DB by processing documents and creating embeddings.
    This should be run whenever the database needs to be regenerated.
    """
    print("Setting up Chroma DB...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file")
        sys.exit(1)
    
    try:
        # Process documents and create embeddings
        print("Processing documents...")
        chunks = process_documents()
        
        if not chunks:
            print("Error: No documents were processed")
            sys.exit(1)
            
        print(f"Creating vector store with {len(chunks)} chunks...")
        store_embeddings(chunks)
        
        print("\n✅ Chroma DB setup completed successfully!")
        print(f"The database has been created in the 'chroma_db' directory.")
        
    except Exception as e:
        print(f"\n❌ Error setting up Chroma DB: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    setup_chroma_db()
