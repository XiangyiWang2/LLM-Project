#!/usr/bin/env python3
import os, json
from pathlib import Path
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    docs = list(load_jsonl("data/raw/hp_corpus.jsonl"))
    texts = [d["text"] for d in docs]
    enc = SentenceTransformer("BAAI/bge-m3")
    embs = enc.encode(texts, batch_size=32, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=True).astype("float32")
    Path("data/index").mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, "data/index/faiss.index")
    with open("data/index/meta.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("Index OK:", len(docs))

if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()
