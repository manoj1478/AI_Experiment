# RAG (Retrieval-Augmented Generation) Project

This project implements a RAG system that combines document retrieval with language model generation to provide context-aware responses to queries.

## Project Structure

```
RAG_Project/
├── config/
│   └── config.py         # Configuration settings
├── data/                 # Store your documents here
├── src/
│   ├── document_processor.py  # Document loading and processing
│   ├── vector_store.py       # Vector storage and similarity search
│   └── rag_system.py         # Main RAG implementation
├── tests/                # Test files
├── .env                 # Environment variables
├── .gitignore          # Git ignore file
├── main.py             # Example usage
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key
   - Adjust other settings as needed

## Usage

1. Place your documents in the `data/` directory

2. Run the example:
   ```bash
   python main.py
   ```

## Features

- Document Processing: Supports PDF, TXT, and other document formats
- Vector Storage: Uses FAISS for efficient similarity search
- Embeddings: Utilizes Sentence Transformers for document embeddings
- LLM Integration: Connects with OpenAI's GPT models for generation

## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies

## License

MIT License