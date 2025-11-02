import json, argparse, re
import faiss, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_meta(p):
    txt=open(p,'r',encoding='utf-8').read().strip()
    return (json.loads(txt) if txt.startswith('[')
            else [json.loads(l) for l in txt.splitlines() if l.strip()])

def gen_answer(model,tok,prompt,max_new_tokens=128):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.2, top_p=0.9, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("[ASSISTANT]")[-1].strip() if "[ASSISTANT]" in text else text

def build_prompt(q, passages):
    cite = "\n".join([f"[{i+1}] {t}" for i,t in enumerate(passages,1)])
    sys = ("You are a helpful assistant. Answer ONLY using the cited passages. "
           "Cite like [1][2]. If unknown, say you don't know.")
    user = f"Question: {q}\n\nPassages:\n{cite}\n\nAnswer (with citations):"
    return f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--index_dir',default='data/index')
    ap.add_argument('--qa_path',default='data/raw/hp_eval_qa.jsonl')
    ap.add_argument('--n',type=int,default=50)
    ap.add_argument('--top_k',type=int,default=20)
    ap.add_argument('--rerank_top',type=int,default=5)
    ap.add_argument('--save_json', default='');
    ap.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct')
    args=ap.parse_args()

    meta = load_meta(f"{args.index_dir}/meta.json")
    idx  = faiss.read_index(f"{args.index_dir}/faiss.index")
    bi   = SentenceTransformer("BAAI/bge-m3")
    ce   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    tok  = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    dtype= torch.float16 if torch.cuda.is_available() else torch.float32
    model= AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
             torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)

    # 读评测问题
    qa = [json.loads(l) for l in open(args.qa_path,'r',encoding='utf-8')]
    qa = qa[:args.n]

    hit_k = hit_rerank = ans_hit = 0
    for ex in qa:
        q, gold = ex['question'], str(ex['answer'])
        qe = bi.encode([q], normalize_embeddings=True).astype('float32')
        D,I = idx.search(qe, args.top_k)
        cands = [meta[i]['text'] for i in I[0]]

        # Top-k 是否包含 gold 关键词（粗评）
        if any(gold.lower() in t.lower() for t in cands):
            hit_k += 1

        # 重排
        scores = ce.predict([(q,t) for t in cands]).tolist()
        order  = np.argsort(scores)[::-1][:args.rerank_top]
        rerank_texts = [cands[i] for i in order]
        if any(gold.lower() in t.lower() for t in rerank_texts):
            hit_rerank += 1

        # 生成并检查答案是否包含关键词（非常粗的 EM 近似）
        ans = gen_answer(model, tok, build_prompt(q, rerank_texts))
        if gold and gold.lower() in ans.lower():
            ans_hit += 1

    n=len(qa)
    print(f"Samples={n}")
    out={"n":n,"Recall@k":hit_k/n,"RerankHit@r":hit_rerank/n,"AnswerHit":ans_hit/n,"top_k":args.top_k,"rerank_top":args.rerank_top,"model":args.model}
    
    import pathlib, json as _j
    if args.save_json:
        pathlib.Path("reports").mkdir(exist_ok=True)
        open(args.save_json,"w").write(_j.dumps(out, indent=2))
        print("saved ->", args.save_json)
    print(f"Recall@{args.top_k}: {hit_k/n:.3f}")
    print(f"RerankHit@{args.rerank_top}: {hit_rerank/n:.3f}")
    print(f"AnswerHit (substring): {ans_hit/n:.3f}")
if __name__=='__main__':
    main()
