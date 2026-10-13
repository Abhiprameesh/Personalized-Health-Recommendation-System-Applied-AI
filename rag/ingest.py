"""
Ingestion script for the RAG knowledge base.
Reads markdown documents in knowledge/, parses metadata frontmatter,
and indexes document chunks into a persistent local ChromaDB collection.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "preventive_wellness_guidelines"


def parse_markdown_with_frontmatter(file_path: Path) -> Dict[str, Any]:
    """Extract YAML frontmatter and content from markdown."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    metadata = {
        "source": "Authoritative Public Health Guidelines",
        "topic": "General Wellness",
        "title": file_path.stem.replace("_", " ").title(),
        "filename": file_path.name
    }

    content = text
    # Parse frontmatter between ---
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if frontmatter_match:
        yaml_content = frontmatter_match.group(1)
        content = frontmatter_match.group(2)
        for line in yaml_content.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()

    return {"metadata": metadata, "content": content}


def chunk_markdown_content(doc: Dict[str, Any], chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Split markdown into meaningful sections by headers or paragraphs."""
    content = doc["content"]
    meta = doc["metadata"]
    sections = re.split(r"\n(?=##?\s)", content)
    
    chunks = []
    chunk_idx = 0
    for sec in sections:
        sec_clean = sec.strip()
        if not sec_clean:
            continue
        
        # If section is very long, split into paragraphs
        if len(sec_clean) > 800:
            paragraphs = sec_clean.split("\n\n")
            for p in paragraphs:
                p_clean = p.strip()
                if len(p_clean) > 50:
                    chunk_idx += 1
                    chunks.append({
                        "id": f"{meta['filename']}_{chunk_idx}",
                        "content": p_clean,
                        "metadata": {
                            "source": meta["source"],
                            "topic": meta["topic"],
                            "title": meta["title"],
                            "filename": meta["filename"]
                        }
                    })
        else:
            chunk_idx += 1
            chunks.append({
                "id": f"{meta['filename']}_{chunk_idx}",
                "content": sec_clean,
                "metadata": {
                    "source": meta["source"],
                    "topic": meta["topic"],
                    "title": meta["title"],
                    "filename": meta["filename"]
                }
            })
            
    return chunks


def ingest_knowledge():
    """Build and populate the local ChromaDB vector database."""
    print(f"Ingesting knowledge guidelines from: {KNOWLEDGE_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Reset or get collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Authoritative Public Health Guidelines for Wellness"}
    )
    
    all_chunks = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        doc = parse_markdown_with_frontmatter(md_file)
        chunks = chunk_markdown_content(doc)
        all_chunks.extend(chunks)
        
    print(f"Parsed {len(all_chunks)} guideline chunks from {len(list(KNOWLEDGE_DIR.glob('*.md')))} files.")
    
    ids = [c["id"] for c in all_chunks]
    documents = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Successfully ingested {len(ids)} chunks into ChromaDB at {CHROMA_DIR}")
    return len(ids)


if __name__ == "__main__":
    ingest_knowledge()
