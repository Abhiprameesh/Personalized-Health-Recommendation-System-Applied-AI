import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import List, Dict, Any, Optional
import chromadb
from rag.ingest import CHROMA_DIR, COLLECTION_NAME, KNOWLEDGE_DIR, ingest_knowledge, parse_markdown_with_frontmatter, chunk_markdown_content

_chroma_client = None
_collection = None


def get_collection():
    """Lazily load or initialize the Chroma collection."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print("ChromaDB storage not found. Running auto-ingestion...")
        ingest_knowledge()

    try:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error accessing ChromaDB collection: {e}. Re-ingesting...")
        ingest_knowledge()
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)

    return _collection


def fallback_keyword_search(query: str, top_k: int = 3, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Robust fallback search directly over knowledge/ files if vector engine is unavailable."""
    results = []
    query_words = set(query.lower().split())
    
    for md_file in KNOWLEDGE_DIR.glob("*.md"):
        doc = parse_markdown_with_frontmatter(md_file)
        if topic and topic.lower() not in doc["metadata"]["topic"].lower():
            continue
        chunks = chunk_markdown_content(doc)
        for chunk in chunks:
            c_text = chunk["content"].lower()
            score = sum(1 for word in query_words if word in c_text)
            if score > 0:
                results.append({
                    "content": chunk["content"],
                    "source": chunk["metadata"]["source"],
                    "topic": chunk["metadata"]["topic"],
                    "title": chunk["metadata"]["title"],
                    "relevance_score": score
                })
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]


def retrieve_guidelines(query: str, top_k: int = 3, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant guideline passages for a given query and optional topic.
    Returns list of dictionaries containing content and metadata.
    """
    try:
        coll = get_collection()
        where_filter = None
        if topic:
            where_filter = {"topic": {"$eq": topic}}
            
        res = coll.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        passages = []
        if res and "documents" in res and res["documents"] and len(res["documents"][0]) > 0:
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            distances = res.get("distances", [[]])[0] if "distances" in res else []
            
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                score = round(1.0 - distances[i], 3) if i < len(distances) and distances[i] is not None else 0.85
                passages.append({
                    "content": doc,
                    "source": meta.get("source", "Authoritative Public Health Guidelines"),
                    "topic": meta.get("topic", "General Wellness"),
                    "title": meta.get("title", "Health Guidance"),
                    "relevance_score": score
                })
            return passages
        else:
            return fallback_keyword_search(query, top_k=top_k, topic=topic)
            
    except Exception as e:
        print(f"ChromaDB retrieval warning ({e}), falling back to direct knowledge search.")
        return fallback_keyword_search(query, top_k=top_k, topic=topic)


if __name__ == "__main__":
    test_res = retrieve_guidelines("walking aerobic exercise 30 minutes", top_k=2)
    print(f"Retrieved {len(test_res)} passages:")
    for r in test_res:
        print(f"[{r['source']}] {r['title']}: {r['content'][:80]}...")
