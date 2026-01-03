import chromadb

# chroma_client = chromadb.PersistentClient(path="./chroma_storage")
# collection = chroma_client.get_or_create_collection("documents")


def get_chroma_collection(collection_name: str = "documents"):
    """
    Get or create a ChromaDB collection with the specified name.
    Args:
        collection_name (str): The name of the collection to retrieve or create.
    Returns:
        Collection: The ChromaDB collection object.
    """
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(collection_name)
    return collection

def add_document_to_collection(collection, embedding, content, metadata, doc_id):
    """
    Add a document, its embedding, and metadata to the specified ChromaDB collection.
    Args:
        collection: The ChromaDB collection object.
        embedding: The embedding vector for the document.
        content: The raw content of the document.
        metadata: Metadata dictionary for the document.
        doc_id: Unique identifier for the document.
    """
    collection.add(
        embeddings=[embedding],
        documents=[content],
        metadatas=[metadata],
        ids=[doc_id]
    )
