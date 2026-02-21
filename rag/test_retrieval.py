import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
INDEX_DIR = "data/index"
MODEL_NAME = "BAAI/bge-m3"
def main():
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    meta_path = os.path.join(INDEX_DIR, "meta.json")
    documents = []
    with open(meta_path, 'r', encoding = 'utf-8') as f:
        documents = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    print("Successfully loading system")
    while True:
        query = input("\n Please Enter Your Problem(Exit By q)")
        if query.lower() == 'q':
            break
        q_embedding = model.encode([query], normalize_embeddings=True)
        q_embedding = np.array(q_embedding).astype('float32')
        D, I  = index.search(q_embedding, 3)
        print("Find Three Possible Answer")
        for i, idx in enumerate(I[0]):
            score = D[0][i]
            if isinstance(documents[idx], dict):
                answer = documents[idx]['text']
            else:
                answer = documents[idx]
            print(f"{i+1} Similarity:{score:.4f}")
            print(f"Answer: {answer[:100]}")
            print("-"*20)
if __name__ == "__main__":
    main()

