"""
Document processing utilities for RAG system.
Handles different document types and prepares them for embedding.
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    def load_document(self, file_path: str) -> List[Dict]:
        """Load and split a document into chunks."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
            
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        """Process multiple documents."""
        all_chunks = []
        for file_path in file_paths:
            chunks = self.load_document(file_path)
            all_chunks.extend(chunks)
        return all_chunks