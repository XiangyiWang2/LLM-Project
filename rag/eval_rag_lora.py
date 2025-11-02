import json, argparse, re, time
import faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def load_meta(p):
    txt = open(p, 'r', encoding='utf-8').read().strip()
    return (json.loads(txt) if txt.startswith('[')
            else [json.loads(l) for l in txt.splitlines() if l.strip()])

def gen_answer(model, tok, prompt, max_new_tokens=128):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 确定性
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("[ASSISTANT]")[-1].strip() if "[ASSISTANT]" in text else text

def build_prompt(q, passages):
    cite = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(passages, 1)])
    sys = ("You are a helpful assistant. Answer ONLY using the cited passages. "
           "Cite like [1][2]. If unknown, say you don't know.")
    user = f"Question: {q}\n\nPassages:\n{cite}\n\nAnswer (with citations):"
    return f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_dir', default='data/index')
    ap.add_argument('--qa_path',   default='data/raw/hp_eval_qa.jsonl')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--rerank_top', type=int, default=5)
    ap.add_argument('--save_json', default=None)
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--peft_path', default=None)
    args = ap.parse_args()

    # 索引与元数据
    meta = load_meta(f"{args.index_dir}/meta.json")
    idx  = faiss.read_index(f"{args.index_dir}/faiss.index")

    # 编码与重排（尽量用 GPU）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bi  = SentenceTransformer("BAAI/bge-m3", device=dev)
    ce  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=dev)

    # 加载基座 + LoRA（推理）
    tok  = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    dtype= torch.float16 if torch.cuda.is_available() else torch.float32
    model= AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=("auto" if torch.cuda.is_available() else None)
    )
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()

    # 读评测问题
    qa = [json.loads(l) for l in open(args.qa_path, 'r', encoding='utf-8')]
    qa = qa[:args.n]

    hit_k = hit_rerank = ans_hit = 0
    t0 = time.time()

    for i, ex in enumerate(tqdm(qa, total=len(qa)), 1):
        q, gold = ex['question'], str(ex['answer'])

        qe = bi.encode(q, normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        qe = qe.reshape(1, -1)
        D, I = idx.search(qe, args.top_k)
        cands = [meta[i]['text'] for i in I[0]]

        if any(gold.lower() in t.lower() for t in cands):
            hit_k += 1

        scores = ce.predict([(q, t) for t in cands], batch_size=min(64, len(cands))).tolist()
        order  = np.argsort(scores)[::-1][:args.rerank_top]
        rerank_texts = [cands[j] for j in order]

        if any(gold.lower() in t.lower() for t in rerank_texts):
            hit_rerank += 1

        ans = gen_answer(model, tok, build_prompt(q, rerank_texts))
        if gold and gold.lower() in ans.lower():
            ans_hit += 1

    n = len(qa)
    result = {
        "n": n,
        "Recall@k": round(hit_k / n, 3),
        "RerankHit@r": round(hit_rerank / n, 3),
        "AnswerHit": round(ans_hit / n, 3),
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "model": args.model,
    }
    if args.peft_path:
        result["peft_path"] = args.peft_path

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
