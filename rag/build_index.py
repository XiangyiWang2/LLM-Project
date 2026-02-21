import json
import os
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
corpus_file = "data/processed/corpus.jsonl"
index_dir = "data/index"
model_name = "BAAI/bge-m3"
def main():
    if not os.path.exists(corpus_file):
        print(f"Can not find {corpus_file}")
        return
    documents = []
    with open(corpus_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    texts = []
    for document in documents:
        texts.append(document['text'])
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Can not download the model, {e}")
    embeddings = model.encode(texts, batch_size = 16, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss.index")
    faiss.write_index(index, index_path)
    meta_path = os.path.join(index_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii = False, indent = 2)
    print(f"In total we have {index.ntotal}")
if __name__ == "__main__":
    main()



            