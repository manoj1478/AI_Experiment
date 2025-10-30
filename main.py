"""
Example usage of the RAG system.
"""
import os
from src.rag_system import RAGSystem

def main():
        

    # Example: Ingest documents
    directory_path = "./data/raw_docs"
    documents = []
    
    # Check if the directory exists and add the files to document list
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                documents.append(entry)

    # Initialize the RAG system
    rag = RAGSystem()
    rag.ingest_documents(documents)
    
    # Example: Generate response
    query = "What is the use of the tag run?"
    response = rag.generate_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()