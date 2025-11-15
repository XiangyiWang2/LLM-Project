import argparse, json, os, numpy as np, sys
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

def read_jsonl(p):
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    docs = list(read_jsonl(args.corpus))
    texts = []
    for r in docs:
        t = r.get("text")
        if isinstance(t,str) and t.strip():
            texts.append(t.strip())

    device = "cuda" if faiss.get_num_gpus()==0 and False else ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES","")!="" else "cpu")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        pass

    model = SentenceTransformer(args.model, device=device)

    embs = []
    for i in tqdm(range(0, len(texts), args.batch), desc="encode"):
        batch = texts[i:i+args.batch]
        E = model.encode(batch, batch_size=args.batch, show_progress_bar=False, normalize_embeddings=True)
        embs.append(E.astype("float32"))
    if not embs:
        raise RuntimeError("没有可编码的文本。")
    X = np.vstack(embs)
    d = X.shape[1]


    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, os.path.join(args.index_dir, "faiss.index"))
    np.save(os.path.join(args.index_dir, "embeddings.npy"), X)
    with open(os.path.join(args.index_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in docs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"encoder": args.model, "ntotal": int(index.ntotal), "dim": d}, f, ensure_ascii=False, indent=2)

    print(f"ntotal: {index.ntotal}")
    print(f"saved -> {args.index_dir}/faiss.index, embeddings.npy, corpus.jsonl, meta.json")

if __name__ == "__main__":
    main()
