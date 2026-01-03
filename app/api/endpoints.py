from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.schema import get_chroma_collection
from app.services.rag import RAGService

router = APIRouter()
ingestion_service = IngestionService()
retrieval_service = RetrievalService(get_chroma_collection("documents"))

@router.post("/documents/")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF or DOCX).
    Args:
        file (UploadFile): The file to be ingested.
    Returns:
        Document: The ingested document's metadata and content.
    """
    try:
        # Pass the file to your ingestion service, which should handle PDF/DOCX parsing
        return ingestion_service.ingest_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/documents/")
async def list_documents():
    """
    List all ingested documents.
    Returns:
        List[Document]: A list of all documents' metadata and content.
    """
    return ingestion_service.list_documents()

@router.get("/search/")
async def search_documents(query: str):
    """
    Search for documents containing the query string in their content.
    Args:
        query (str): The search query string.
    Returns:
        List[Document]: A list of matching documents.
    """
    results = retrieval_service.search_documents(query)
    return results

@router.post("/generate/")
async def generate_response(query: str):
    """
    Generate a response to a query using RAG (Retrieval-Augmented Generation).
    """
    MODEL_PATH = r"C:\\Users\\Vinayaka\\OneDrive\\Documents\\Soujanya\\New folder\\rag-backend\\models_new\\llama-2-7b-chat.Q4_K_M.gguf"
    rag_service = RAGService(model_path=MODEL_PATH)
    response = rag_service.generate_response(query)
    return {"answer": response}