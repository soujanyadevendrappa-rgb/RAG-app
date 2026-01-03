from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from app.services.schema import get_chroma_collection


class RAGService:
    def __init__(self, model_path, retrieval_service=None, ingestion_service=None):
        """
        Retrieval-Augmented Generation service using ChromaDB and llama-cpp-python.
        """

        # -------------------------------
        # 1. Load Chroma collection
        # -------------------------------
        self.collection = get_chroma_collection("documents")

        # -------------------------------
        # 2. Load embedding model
        # -------------------------------
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # -------------------------------
        # 3. Load LLaMA model optimally
        # -------------------------------
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,        # increase context window
            n_threads=8,       # set based on your CPU cores
            n_batch=512,       # speeds up inference
            use_mmap=True,     # faster load
            verbose=False
        )

        print("LLaMA model loaded successfully. n_ctx =", self.llm.n_ctx())

        self.retrieval_service = retrieval_service
        self.ingestion_service = ingestion_service

    def generate_response(self, query, top_k=3):
        """
        Generate a response based on semantic search + LLaMA generation.
        """

        # 1. Embed query
        query_embedding = self.embedder.encode([query])[0]

        # 2. Retrieve docs
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"]
        )
        documents = results["documents"][0] if results["documents"] else []

        # 3. Combine docs into context
        context = self._prepare_context(documents)

        # 4. Generate final answer
        return self._generate_answer(query, context)

    def _prepare_context(self, documents):
        """
        Combine multiple retrieved documents into one context string.
        """
        return "\n".join(documents)

    def _generate_answer(self, query, context):
        """
        Generate answer using local LLaMA model safely without exceeding n_ctx.
        """

        # 1. Base prompt without injecting context yet
        prompt_template = (
            "You are a helpful assistant. Use the following context to answer the user's question.\n\n"
            "Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        # 2. Get model's max context size
        n_ctx = self.llm.n_ctx()
        max_gen = 512  # space for generated output
        max_prompt_tokens = n_ctx - max_gen - 50

        # 3. Tokenize context to keep prompt safe
        encoded = self.llm.tokenize(context.encode("utf-8"))

        # Trim context if necessary
        if len(encoded) > max_prompt_tokens:
            encoded = encoded[:max_prompt_tokens]
            context = self.llm.detokenize(encoded).decode("utf-8", errors="ignore")

        # 4. Build final, safe prompt
        prompt = prompt_template.format(context=context)

        # 5. Call llama.cpp model
        response = self.llm(
            prompt=prompt,
            max_tokens=max_gen,
            temperature=0.2,
            top_p=0.9,
            stop=["</s>"]
        )

        return response["choices"][0]["text"].strip()
