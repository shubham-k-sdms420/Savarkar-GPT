"""
Data Ingestion Pipeline.

Orchestrates the full ingestion flow:
  Load JSON -> Chunk Documents -> Embed -> Upsert to Pinecone

Run this module to populate the vector database with the knowledge base.
"""

from pathlib import Path

from app.chunking.chunker import DocumentChunker
from app.embeddings.embedder import Embedder
from app.vectorstore.pinecone_store import PineconeStore
from app.config.settings import settings


class IngestionPipeline:
    """End-to-end data ingestion: JSON -> chunks -> embeddings -> Pinecone."""

    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: PineconeStore | None = None,
    ):
        self.chunker = chunker or DocumentChunker()

        # Share a single embedder instance
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or PineconeStore(embedder=self.embedder)

    def run(self, data_dir: Path | None = None) -> dict:
        """
        Execute the full ingestion pipeline.

        Args:
            data_dir: Path to directory containing JSON files.
                      Defaults to settings.DATA_DIR.

        Returns:
            Summary dict with stats about the ingestion.
        """
        data_dir = data_dir or settings.DATA_DIR

        print("=" * 60)
        print("SAVARKAR GPT - DATA INGESTION PIPELINE")
        print("=" * 60)

        settings.print_config()

        # Validate settings
        errors = settings.validate()
        if errors:
            for err in errors:
                print(f"  ERROR: {err}")
            raise RuntimeError("Configuration errors found. Fix .env and retry.")

        # Step 1: Chunk documents
        print("\n--- STEP 1: Chunking Documents ---")
        chunks = self.chunker.chunk_all_documents(data_dir)

        if not chunks:
            raise RuntimeError("No chunks produced. Check your JSON files.")

        # Step 2 & 3: Embed + Upsert (handled inside vector_store)
        print("\n--- STEP 2: Embedding & Upserting to Pinecone ---")
        total_upserted = self.vector_store.upsert_documents(chunks)

        # Step 4: Verify
        print("\n--- STEP 3: Verification ---")
        stats = self.vector_store.get_index_stats()
        print(f"  Index stats: {stats}")

        summary = {
            "data_dir": str(data_dir),
            "total_chunks": len(chunks),
            "total_upserted": total_upserted,
            "index_stats": stats,
        }

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print(f"  Total chunks created:  {len(chunks)}")
        print(f"  Total vectors upserted: {total_upserted}")
        print(f"  Vectors in index:      {stats.get('total_vectors', 'N/A')}")
        print("=" * 60)

        return summary


def run_ingestion():
    """Convenience function to run ingestion from command line."""
    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    run_ingestion()
