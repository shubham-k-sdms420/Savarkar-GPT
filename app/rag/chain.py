"""
RAG Chain Module.

Orchestrates the full Retrieval-Augmented Generation pipeline:
  Query -> Retrieve from Pinecone -> Format Context -> Generate via Gemini

This is the core module that ties everything together.
Plug-and-play: Each component (retriever, LLM) can be swapped independently.
"""

from langchain_core.documents import Document

from app.vectorstore.pinecone_store import PineconeStore
from app.llm.gemini import GeminiLLM
from app.embeddings.embedder import Embedder
from app.config.settings import settings


class RAGChain:
    """RAG pipeline: Retrieve + Augment + Generate."""

    def __init__(
        self,
        vector_store: PineconeStore | None = None,
        llm: GeminiLLM | None = None,
        top_k: int | None = None,
    ):
        # Shared embedder instance across vectorstore and queries
        self._embedder = Embedder()
        self.vector_store = vector_store or PineconeStore(embedder=self._embedder)
        self.llm = llm or GeminiLLM()
        self.top_k = top_k or settings.TOP_K_RESULTS

    def _format_context(self, documents: list[Document]) -> str:
        """
        Format retrieved documents into a context string for the LLM.

        Each passage includes its source metadata for citation.
        """
        if not documents:
            return "No relevant passages found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("title", "Unknown")
            author = doc.metadata.get("author", "Unknown")
            chapter = doc.metadata.get("chapter_title", "")
            score = doc.metadata.get("score", 0.0)

            header = f"[Passage {i}] From: \"{source}\" by {author}"
            if chapter:
                header += f" | Chapter: {chapter}"
            header += f" | Relevance: {score:.2f}"

            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
    ) -> dict:
        """
        Full RAG pipeline: retrieve context and generate answer.

        Args:
            question: The user's question.
            top_k: Number of passages to retrieve.
            filter_dict: Optional metadata filter for Pinecone.

        Returns:
            Dict with 'answer', 'sources', and 'context' keys.
        """
        top_k = top_k or self.top_k

        # Step 1: Retrieve relevant documents from Pinecone
        retrieved_docs = self.vector_store.query(
            query_text=question,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        # Step 2: Format context from retrieved documents
        context = self._format_context(retrieved_docs)

        # Step 3: Generate answer using Gemini with grounded context
        answer = self.llm.generate(query=question, context=context)

        # Step 4: Extract source citations
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "chapter": doc.metadata.get("chapter_title", ""),
                "relevance_score": doc.metadata.get("score", 0.0),
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_passages_retrieved": len(retrieved_docs),
        }

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Retrieve documents without generating an answer.
        Useful for debugging retrieval quality.
        """
        return self.vector_store.query(
            query_text=question,
            top_k=top_k or self.top_k,
        )
