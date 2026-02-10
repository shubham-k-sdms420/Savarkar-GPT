"""
Document Chunking Module.

Loads JSON knowledge base files and splits them into chunks
with rich metadata for vector storage.

ALL parameters (chunk_size, overlap, separators) come from settings/.env.
Plug-and-play: Swap chunking strategy by changing this module.
"""

import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config.settings import settings


class DocumentChunker:
    """Chunks documents from JSON knowledge base files."""

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.separators = settings.get_chunk_separators()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
            is_separator_regex=False,
        )

    def load_json_file(self, file_path: Path) -> dict:
        """Load a single JSON knowledge base file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def chunk_single_document(self, file_path: Path) -> list[Document]:
        """
        Load and chunk a single JSON document.

        Each chunk gets metadata:
          - source: filename
          - title: book title
          - author: book author
          - category: book category (history, biography, etc.)
          - chapter_number: chapter it belongs to
          - chapter_title: chapter title
          - chunk_index: position within the chapter
        """
        data = self.load_json_file(file_path)
        metadata_base = data.get("metadata", {})
        chapters = data.get("chapters", [])

        all_chunks = []

        for chapter in chapters:
            chapter_num = chapter.get("chapter_number", 0)
            chapter_title = chapter.get("chapter_title", "Unknown")
            paragraphs = chapter.get("paragraphs", [])

            if not paragraphs:
                continue

            # Join paragraphs into chapter text
            chapter_text = "\n\n".join(paragraphs)

            if not chapter_text.strip():
                continue

            # Split chapter text into chunks
            chunks = self.text_splitter.split_text(chapter_text)

            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_path.name,
                        "title": metadata_base.get("title", "Unknown"),
                        "author": metadata_base.get("author", "Unknown"),
                        "category": metadata_base.get("category", "unknown"),
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_title,
                        "chunk_index": i,
                    },
                )
                all_chunks.append(doc)

        return all_chunks

    def chunk_all_documents(self, data_dir: Path | None = None) -> list[Document]:
        """
        Load and chunk ALL JSON documents from the data directory.
        Skips files starting with '_' (index files, reports).
        """
        data_dir = data_dir or settings.DATA_DIR
        all_chunks = []

        json_files = sorted(
            f for f in data_dir.glob("*.json") if not f.name.startswith("_")
        )

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")

        print(f"\nChunking {len(json_files)} documents from {data_dir}")
        print(f"  Chunk size: {self.chunk_size} | Overlap: {self.chunk_overlap}")

        for file_path in json_files:
            chunks = self.chunk_single_document(file_path)
            print(f"  {file_path.name}: {len(chunks)} chunks")
            all_chunks.append(chunks)

        # Flatten
        flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

        print(f"\nTotal chunks: {len(flat_chunks)}")
        return flat_chunks
