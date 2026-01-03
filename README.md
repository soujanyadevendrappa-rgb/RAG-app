# Document Ingestion and Retrieval-Augmented Generation (RAG) Q&A System

## Overview
This project implements a Document Ingestion and Retrieval-Augmented Generation (RAG) Q&A system using FastAPI. The application allows users to ingest documents and retrieve information through a question-answering interface. It leverages ChromaDB for vector storage and OpenAI for answer generation.

## Project Structure
```
rag-backend
├── app
│   ├── main.py               # Entry point of the FastAPI application
│   ├── api
│   │   └── endpoints.py      # API endpoints for document ingestion and retrieval
│   ├── models
│   │   └── document.py       # Data model for documents
│   ├── services
│   │   ├── ingestion.py       # Document ingestion logic
│   │   ├── retrieval.py       # Document retrieval logic
│   │   ├── rag.py             # Retrieval-Augmented Generation logic
│   │   └── schema.py          # ChromaDB connection and schema logic
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-backend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## Usage
- **Ingest Document**: Send a POST request to `/documents/` with a PDF or DOCX file using `multipart/form-data` (field name: `file`).
- **List Documents**: Send a GET request to `/documents/` to list all ingested documents.
- **Search Documents**: Send a GET request to `/search/?query=your_query` to search for documents semantically.
- **Generate Response**: Send a POST request to `/generate/` with a `query` parameter to get a generated response based on the ingested documents and semantic search.

## Technologies Used
- **FastAPI** for building the API
- **ChromaDB** for vector storage and semantic search
- **Sentence Transformers** for embedding generation