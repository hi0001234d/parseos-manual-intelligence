"""
engine.py — Unified LlamaIndex Engine for ParseOS SOP Engine (v2.0)
===================================================================
Consolidates text splitting, embedding generation, vector database storage,
and semantic query retrieval (Stages 2-5) using LlamaIndex.

v2.0 Architectural Updates
--------------------------
- Ingestion operates page-by-page. Each page creates a LlamaIndex Document.
  SentenceSplitter node parser propagates parent page metadata (page_num,
  has_visual_content, visual_confidence, image_path) into every chunk node.
- SearchResult exposes has_visual_content, visual_confidence, and image_path
  for Stage 6a conditional visual processing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)
from src.pdf_parser import extract_text_with_metadata, get_page_count


# ── SearchResult Dataclass ──────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    Represents a matching chunk retrieved from the vector database.
    """
    chunk_text:         str
    manual_name:        str
    chunk_index:        int
    distance:           float
    score:              float = 0.0
    page_num:           int = -1
    has_visual_content: bool = False
    visual_confidence:  float = 0.0
    image_path:         str = ""
    is_ocr:             bool = False
    similarity:         float = field(init=False)

    def __post_init__(self) -> None:
        self.similarity = round(1.0 - self.distance, 4)

    def __str__(self) -> str:
        return (
            f"[{self.manual_name} | chunk {self.chunk_index} | page {self.page_num}] "
            f"sim={self.similarity:.3f} | visual={self.has_visual_content} ({self.visual_confidence})\n"
            f"{self.chunk_text[:200]}…"
        )


# ── SearchEngine Class ────────────────────────────────────────────────────────

class SearchEngine:
    """
    Unified LlamaIndex-backed search and ingestion engine.
    Manages text ingestion pipeline and similarity search queries.
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = CHROMA_COLLECTION,
        embed_model_name: str = EMBED_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._chroma_path      = chroma_path
        self._collection_name  = collection_name
        self._embed_model_name = embed_model_name
        self._chunk_size       = chunk_size
        self._chunk_overlap    = chunk_overlap

        # Lazy initialized components
        self._embed_model   = None
        self._chroma_client = None
        self._collection     = None
        self._vector_store   = None
        self._index          = None

    # ── Lazy initialization methods ──────────────────────────────────────────

    def _get_embed_model(self):
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._embed_model_name,
                embed_batch_size=64,
            )
        return self._embed_model

    def _get_chroma_collection(self):
        if self._collection is None:
            import chromadb
            os.makedirs(self._chroma_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "description": "ParseOS Industrial Manual Knowledge Base",
                    "hnsw:space": "cosine",
                },
            )
        return self._collection

    def _get_vector_store(self):
        if self._vector_store is None:
            from llama_index.vector_stores.chroma import ChromaVectorStore
            self._vector_store = ChromaVectorStore(
                chroma_collection=self._get_chroma_collection()
            )
        return self._vector_store

    def _get_index(self):
        if self._index is None:
            from llama_index.core import VectorStoreIndex, StorageContext
            storage_context = StorageContext.from_defaults(
                vector_store=self._get_vector_store()
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._get_vector_store(),
                storage_context=storage_context,
                embed_model=self._get_embed_model(),
            )
        return self._index

    # ── Ingestion (Stages 2-4) ───────────────────────────────────────────────

    def ingest_manual(
        self,
        pdf_path: str,
        force: bool = False,
    ) -> int:
        """
        Splits PDF text page-by-page, generates vector embeddings, and stores them in ChromaDB.
        """
        from llama_index.core import Document
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter

        manual_name = Path(pdf_path).stem

        # Skip if already exists (unless forced)
        if not force and self.manual_exists(manual_name):
            count = self.get_manual_chunk_count(manual_name)
            print(f"  [SearchEngine] '{manual_name}' already ingested ({count} chunks).")
            return count

        print(f"  [SearchEngine] Extracting PDF text (text-only v2.0): {pdf_path}")
        pages_count = get_page_count(pdf_path)
        pages_data  = extract_text_with_metadata(pdf_path)

        self._delete_manual_chunks(manual_name)

        documents = []
        for p in pages_data:
            documents.append(
                Document(
                    text=p["text"],
                    metadata={
                        "manual":             manual_name,
                        "source_path":        str(pdf_path),
                        "page_count":         pages_count,
                        "page":               p["page_num"],
                        "has_visual_content": p.get("has_visual_content", False),
                        "visual_confidence":  p.get("visual_confidence", 0.0),
                        "image_path":         p.get("image_path", ""),
                        "ocr_used":           p.get("ocr_used", False),
                    },
                )
            )

        splitter = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            paragraph_separator="\n\n",
        )

        pipeline = IngestionPipeline(
            transformations=[
                splitter,
                self._get_embed_model(),
            ],
            vector_store=self._get_vector_store(),
        )

        print(f"  [SearchEngine] Running IngestionPipeline for '{manual_name}' …")
        nodes = pipeline.run(documents=documents, show_progress=False)

        # Patch metadata indexes for backward compatibility checking
        for i, node in enumerate(nodes):
            node.metadata.setdefault("manual", manual_name)
            node.metadata.setdefault("chunk_index", i)

        total = self._get_chroma_collection().count()
        print(f"  [SearchEngine] Ingested {len(nodes)} nodes. Total in DB: {total}")

        self._index = None
        return len(nodes)

    # ── Retrieval (Stage 5) ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filter_manual: str | None = None,
    ) -> list[SearchResult]:
        """
        Queries LlamaIndex VectorStoreIndex for relevant document nodes.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        total = self.total_chunks()
        if total == 0:
            raise RuntimeError("The knowledge base is empty. Ingest manuals first.")

        top_k = min(top_k, total)

        filters = None
        if filter_manual:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
            filters = MetadataFilters(
                filters=[MetadataFilter(key="manual", value=filter_manual)]
            )

        retriever = self._get_index().as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )

        nodes = retriever.retrieve(query)

        results: list[SearchResult] = []
        for node_with_score in nodes:
            node  = node_with_score.node
            score = float(node_with_score.score or 0.0)
            meta  = node.metadata or {}
            distance = max(0.0, 1.0 - score)

            results.append(
                SearchResult(
                    chunk_text=node.get_content(metadata_mode="none"),
                    manual_name=meta.get("manual", "unknown"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=distance,
                    score=score,
                    page_num=int(meta.get("page", -1)),
                    has_visual_content=bool(meta.get("has_visual_content", False)),
                    visual_confidence=float(meta.get("visual_confidence", 0.0)),
                    image_path=meta.get("image_path", ""),
                    is_ocr=bool(meta.get("ocr_used", False)),
                )
            )

        # Sort results by distance (closest first)
        results.sort(key=lambda r: r.distance)
        return results

    def search_combined_text(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filter_manual: str | None = None,
    ) -> str:
        """Returns joined matching chunk texts for LLM injection."""
        results = self.search(query, top_k=top_k, filter_manual=filter_manual)
        return "\n\n".join(r.chunk_text for r in results)

    # ── Utility Methods ──────────────────────────────────────────────────────

    def total_chunks(self) -> int:
        try:
            return self._get_chroma_collection().count()
        except Exception:
            return 0

    def list_manuals(self) -> list[str]:
        try:
            col    = self._get_chroma_collection()
            result = col.get(include=["metadatas"], limit=100_000)
            names  = sorted({m["manual"] for m in result["metadatas"] if "manual" in m})
            return names
        except Exception:
            return []

    def manual_exists(self, manual_name: str) -> bool:
        return self.get_manual_chunk_count(manual_name) > 0

    def get_manual_chunk_count(self, manual_name: str) -> int:
        try:
            col    = self._get_chroma_collection()
            result = col.get(where={"manual": manual_name}, include=["metadatas"])
            return len(result["ids"])
        except Exception:
            return 0

    def get_manual_text(self, manual_name: str) -> str:
        """
        Returns concatenated text of all stored chunks for a manual.
        Used by the --build-vocab backfill path in ingest_all.py to extract
        technical vocabulary without re-parsing the original PDF.
        Returns empty string if the manual is not found or on any error.
        """
        try:
            col = self._get_chroma_collection()
            result = col.get(
                where={"manual": manual_name},
                include=["documents"],
            )
            docs = result.get("documents") or []
            return " ".join(docs)
        except Exception:
            return ""

    def _delete_manual_chunks(self, manual_name: str) -> None:
        try:
            col      = self._get_chroma_collection()
            existing = col.get(where={"manual": manual_name}, include=[])
            ids      = existing.get("ids", [])
            if ids:
                col.delete(ids=ids)
                print(f"  [SearchEngine] Deleted {len(ids)} old nodes for '{manual_name}'.")
        except Exception:
            pass


# ── Module-Level Convenience Functions ────────────────────────────────────────

def search_manual(
    query: str,
    top_k: int = TOP_K_RESULTS,
    filter_manual: str | None = None,
) -> tuple[list[str], list[dict], list[float]]:
    """Convenience shortcut matching standard project verification."""
    engine  = SearchEngine()
    results = engine.search(query, top_k=top_k, filter_manual=filter_manual)

    chunks    = [r.chunk_text  for r in results]
    metas     = [{"manual": r.manual_name, "chunk_index": r.chunk_index, "page": r.page_num, "has_visual": r.has_visual_content} for r in results]
    distances = [r.distance    for r in results]
    return chunks, metas, distances