# app/rag/ingestor.py
import json
import os
import time
import chromadb
from rank_bm25 import BM25Okapi
#from langchain_groq import ChatGroq
#from langchain_core.messages import HumanMessage
from app.core.config import settings
from app.core.prompts import CONTEXTUAL_RETRIEVAL_PROMPT
from app.rag.embedder import embedder
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Ingestor:
    """
    Ingestion pipeline — runs ONCE to build the knowledge base.
    
    What happens here:
    1. Load raw support Q&A pairs from JSONL
    2. For each doc, call LLM to generate contextual prefix (Anthropic's technique)
    3. Prepend context to chunk → richer embedding
    4. Embed enriched chunks → store in ChromaDB
    5. Also build BM25 index → for hybrid search at query time
    
    Why contextual retrieval?
    Naive chunking loses document-level context. A chunk saying 
    "Yes, you can do that within 30 days" is meaningless without knowing 
    WHAT you can do. Contextual retrieval prepends: 
    "This chunk is about the return policy for orders. Yes, you can do that within 30 days."
    Retrieval accuracy improves ~49% on average (Anthropic's benchmark).
    """

    def __init__(self):
     self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
     self.collection = self.chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    def ingest(self, jsonl_path: str, sample_size: int = 200):
        logger.info(f"Loading dataset from {jsonl_path}")
        rows = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                rows.append(json.loads(line))

        logger.info(f"Loaded {len(rows)} rows. Starting ingestion...")

        # Check if already partially ingested — skip existing docs
        existing_ids = set(self.collection.get()["ids"])
        logger.info(f"Already indexed: {len(existing_ids)} docs")

        documents = []
        metadatas = []
        ids = []
        bm25_corpus = []

        for i, row in enumerate(rows):
            doc_id = f"doc_{i}"
            
            # Skip already indexed docs — resume capability
            if doc_id in existing_ids:
                logger.info(f"Skipping {doc_id} — already indexed")
                continue

            if i % 10 == 0:
                logger.info(f"Processing {i}/{len(rows)}...")

            # Skip LLM call entirely — use metadata as context directly
            # This is a lightweight version of contextual retrieval
            # Still better than naive chunking — adds category/intent signal
            context = (
                f"This support interaction is about {row.get('category', 'general')} "
                f"specifically addressing {row.get('intent', 'customer query').replace('_', ' ')}."
            )

            enriched_doc = (
                f"{context}\n\n"
                f"Q: {row['instruction']}\n"
                f"A: {row['response']}"
            )

            documents.append(enriched_doc)
            metadatas.append({
                "category": row.get("category", ""),
                "intent": row.get("intent", ""),
                "original_question": row.get("instruction", ""),
                "original_answer": row.get("response", "")
            })
            ids.append(doc_id)
            bm25_corpus.append(enriched_doc.lower().split())

        if not documents:
            logger.info("Nothing new to index.")
            return 0

        # Embed all at once
        logger.info("Generating embeddings...")
        embeddings = embedder.embed(documents)

        # Store in ChromaDB
        logger.info("Storing in ChromaDB...")
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        # Build BM25
        logger.info("Building BM25 index...")
        # Include existing docs in BM25 if any
        all_docs = documents
        bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
        bm25_data = {"bm25": bm25, "corpus": all_docs}
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        logger.info(f"✅ Ingestion complete. {len(documents)} docs indexed.")
        return len(documents)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    ingestor = Ingestor()
    ingestor.ingest("data/raw/support_dataset.jsonl", sample_size=500)