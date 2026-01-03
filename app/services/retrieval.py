from sentence_transformers import SentenceTransformer

class RetrievalService:
    def __init__(self, collection):
        self.collection = collection
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def search_documents(self, query, top_k=5):
        query_embedding = self.embedder.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        documents = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            documents.append({
                "title": meta.get("title", meta.get("filename", "")),
                "content": doc,
                "metadata": meta
            })
        return documents