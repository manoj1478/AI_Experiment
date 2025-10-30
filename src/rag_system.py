"""
RAG (Retrieval-Augmented Generation) system implementation.
Combines document retrieval with LLM generation.
"""
from typing import List, Dict
import os
from dotenv import load_dotenv
import google.generativeai as genai
from .vector_store import VectorStore
from .document_processor import DocumentProcessor

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        # Configure Gemini / Google Generative API client.
        # The code will look for GEMINI_API_KEY, then GOOGLE_API_KEY, then OPENAI_API_KEY
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # leave unconfigured; downstream calls will raise a helpful error
            pass
        
    def ingest_documents(self, file_paths: List[str]):
        """Process and store documents in the vector store."""
        documents = self.document_processor.process_documents(file_paths)
        self.vector_store.add_documents(documents)

        
    def generate_response(self, query: str) -> str:
        """Generate a response using Retrieval-Augmented Generation (RAG).
        This method performs the following steps:
        1. Retrieves relevant documents from the vector store based on the input query
        2. Combines the retrieved documents into a context
        3. Uses OpenAI's ChatCompletion to generate a response considering both query and context
        Args:
            query (str): The user's input question or query
        Returns:
            str: The generated response from the language model based on retrieved context
        Example:
            >>> rag = RAGSystem()
            >>> response = rag.generate_response("What is machine learning?")
            >>> print(response)
        """

        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query)
        
        # Prepare context from retrieved documents
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate response using OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Call Gemini / Google Generative AI chat. We use a best-effort, flexible
        # call and robustly extract text from common response shapes. The
        # environment variable LLM_MODEL can be used to select a Gemini model
        # (for example, "gemini-1.5", "gemini-1.0", or other provider-specific id).
        model = os.getenv("LLM_MODEL", "gemini-1.0")

        resp = genai.chat.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_output_tokens=2000,
        )

        # Flexible extraction of the reply text from common shapes returned by
        # various Gemini/GenAI client versions. Try several possible fields.
        # 1) If resp has attribute 'content' or 'message' with a 'content' field
        try:
            # dict-like
            if isinstance(resp, dict):
                if "content" in resp and isinstance(resp["content"], str):
                    return resp["content"]
                if "candidates" in resp and resp["candidates"]:
                    candidate = resp["candidates"][0]
                    if isinstance(candidate, dict):
                        return candidate.get("content") or candidate.get("message") or str(candidate)

            # object-like
            if hasattr(resp, "message"):
                msg = resp.message
                if isinstance(msg, dict):
                    return msg.get("content") or str(msg)
                # try attribute access
                if hasattr(msg, "content"):
                    return msg.content

            if hasattr(resp, "output"):
                out = resp.output
                # many SDKs put text at output[0]['content'][0]['text']
                try:
                    return out[0]["content"][0]["text"]
                except Exception:
                    pass

            # Last resort: convert to string
            return str(resp)
        except Exception as e:
            # Surface an informative error rather than failing silently
            raise RuntimeError(f"Failed to extract text from Gemini response: {e}")