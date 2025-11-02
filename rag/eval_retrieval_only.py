import json, argparse, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_meta(p):
    txt = open(p,'r',encoding='utf-8').read().strip()
    return (json.loads(txt) if txt.startswith('[')
            else [json.loads(l) for l in txt.splitlines() if l.strip()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_dir',default='data/index')
    ap.add_argument('--qa_path', default='data/raw/hp_eval_qa.jsonl')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--rerank_top', type=int, default=5)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = load_meta(f"{args.index_dir}/meta.json")
    idx  = faiss.read_index(f"{args.index_dir}/faiss.index")
    bi   = SentenceTransformer("BAAI/bge-m3", device=device)
    ce   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    qa = [json.loads(l) for l in open(args.qa_path,'r',encoding='utf-8') if l.strip()][:args.n]

    hit_k = hit_r = 0
    for ex in qa:
        q, gold = ex['question'], str(ex['answer'])
        qe = bi.encode([q], normalize_embeddings=True).astype('float32')
        D,I = idx.search(qe, args.top_k)
        cands = [meta[i]['text'] for i in I[0]]

        if any(gold.lower() in t.lower() for t in cands):
            hit_k += 1

        scores = ce.predict([(q,t) for t in cands]).tolist()
        order  = np.argsort(scores)[::-1][:args.rerank_top]
        rerank = [cands[i] for i in order]
        if any(gold.lower() in t.lower() for t in rerank):
            hit_r += 1

    n = len(qa)
    print(f"Samples={n}")
    print(f"Recall@{args.top_k}: {hit_k/n:.3f}")
    print(f"RerankHit@{args.rerank_top}: {hit_r/n:.3f}")

if __name__ == '__main__':
    main()
