"""
Embedding Module.

Wraps sentence-transformers model for generating embeddings.
ALL parameters (model name, batch size) come from settings/.env.

Plug-and-play: Swap embedding model by changing EMBEDDING_MODEL_NAME in .env.
"""

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from app.config.settings import settings


class Embedder:
    """Generates embeddings using sentence-transformers."""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"  Dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """
        Embed a list of LangChain Document objects.

        Args:
            documents: List of Document objects.

        Returns:
            List of embedding vectors.
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
