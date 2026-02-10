"""
Pinecone Vector Store Module.

Handles all Pinecone operations: index creation, upserting, querying.
ALL parameters (cloud, region, metric, batch sizes) come from settings/.env.

Plug-and-play: Swap to another vector DB by replacing this module
with the same interface (upsert_documents, query).
"""

import hashlib
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

from app.config.settings import settings
from app.embeddings.embedder import Embedder


class PineconeStore:
    """Manages Pinecone vector database operations."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index_name = settings.PINECONE_INDEX_NAME
        self._client = None
        self._index = None

    @property
    def client(self) -> Pinecone:
        """Lazy-init Pinecone client."""
        if self._client is None:
            self._client = Pinecone(api_key=settings.PINECONE_API_KEY)
        return self._client

    @property
    def index(self):
        """Lazy-init Pinecone index."""
        if self._index is None:
            self._ensure_index_exists()
            self._index = self.client.Index(self.index_name)
        return self._index

    def _ensure_index_exists(self):
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.client.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.client.create_index(
                name=self.index_name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
            print(f"  Index '{self.index_name}' created successfully.")
        else:
            print(f"Pinecone index '{self.index_name}' already exists.")

    @staticmethod
    def _generate_id(text: str, metadata: dict) -> str:
        """Generate a deterministic ID for a chunk to avoid duplicates."""
        unique_str = (
            f"{metadata.get('source', '')}_"
            f"{metadata.get('chapter_number', '')}_"
            f"{metadata.get('chunk_index', '')}_"
            f"{text[:100]}"
        )
        return hashlib.md5(unique_str.encode()).hexdigest()

    def upsert_documents(self, documents: list[Document]) -> int:
        """
        Embed and upsert documents into Pinecone.

        Args:
            documents: List of LangChain Document objects with metadata.

        Returns:
            Total number of vectors upserted.
        """
        batch_size = settings.PINECONE_UPSERT_BATCH_SIZE
        text_limit = settings.PINECONE_METADATA_TEXT_LIMIT

        print(f"\nEmbedding {len(documents)} chunks...")
        embeddings = self.embedder.embed_documents(documents)

        print(f"Upserting {len(documents)} vectors to Pinecone index '{self.index_name}'...")

        total_upserted = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            vectors = []
            for doc, embedding in zip(batch_docs, batch_embeddings):
                vec_id = self._generate_id(doc.page_content, doc.metadata)

                # Pinecone metadata must be flat (no nested dicts/lists)
                metadata = {
                    "text": doc.page_content[:text_limit],
                    "source": str(doc.metadata.get("source", "")),
                    "title": str(doc.metadata.get("title", "")),
                    "author": str(doc.metadata.get("author", "")),
                    "category": str(doc.metadata.get("category", "")),
                    "chapter_number": int(doc.metadata.get("chapter_number", 0)),
                    "chapter_title": str(doc.metadata.get("chapter_title", "")),
                    "chunk_index": int(doc.metadata.get("chunk_index", 0)),
                }

                vectors.append({
                    "id": vec_id,
                    "values": embedding,
                    "metadata": metadata,
                })

            self.index.upsert(vectors=vectors)
            total_upserted += len(vectors)

            if (i // batch_size) % 10 == 0:
                print(f"  Upserted {total_upserted}/{len(documents)} vectors...")

        print(f"  Upsert complete: {total_upserted} vectors.")
        return total_upserted

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """
        Query Pinecone for similar documents.

        Args:
            query_text: The user's query string.
            top_k: Number of results to return.
            filter_dict: Optional metadata filter.

        Returns:
            List of LangChain Document objects with score in metadata.
        """
        top_k = top_k or settings.TOP_K_RESULTS

        query_embedding = self.embedder.embed_query(query_text)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )

        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")

            doc = Document(
                page_content=text,
                metadata={
                    **metadata,
                    "score": match.get("score", 0.0),
                },
            )
            documents.append(doc)

        return documents

    def get_index_stats(self) -> dict:
        """Get statistics about the Pinecone index."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "namespaces": dict(stats.get("namespaces", {})),
        }

    def delete_all(self):
        """Delete all vectors from the index. Use with caution."""
        self.index.delete(delete_all=True)
        print(f"All vectors deleted from index '{self.index_name}'.")
