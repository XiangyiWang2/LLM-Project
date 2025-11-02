#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import faiss, numpy as np
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_index(index_dir):
    idx = faiss.read_index(str(Path(index_dir)/"faiss.index"))
    meta_path = Path(index_dir)/"meta.json"
    content = meta_path.read_text(encoding="utf-8").strip()
    if not content:
        meta = []
    elif content.lstrip().startswith("["):
        meta = json.loads(content)  # JSON array
    else:
        meta = [json.loads(line) for line in content.splitlines() if line.strip()]  # JSONL
    return idx, meta

def build_prompt(q, passages):
    cite_text = "\n".join([f"[{i+1}] {t}" for i,(_,t) in enumerate(passages)])
    sys = ("You are a helpful assistant. Answer using ONLY the information from the cited passages. "
           "Cite like [1][2] and avoid hallucination. If unknown, say you don't know.")
    user = f"Question: {q}\n\nPassages:\n{cite_text}\n\nAnswer (with citations):"
    return f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--rerank_top", type=int, default=5)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    print("Loading retrievers...")
    bi = SentenceTransformer("BAAI/bge-m3")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index, meta = load_index(args.index_dir)

    print("Loading LLM...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if args.load_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=True, load_in_4bit=True
        )
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )

    def answer(q):
        if not q.strip():
            return "", ""
        qe = bi.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(qe, args.top_k)
        cand_idxs = I[0].tolist()
        cand_texts = [meta[i]["text"] for i in cand_idxs]

        scores = ce.predict([(q,t) for t in cand_texts]).tolist()
        order = np.argsort(scores)[::-1][:args.rerank_top]
        chosen = [(cand_idxs[i], cand_texts[i]) for i in order]

        prompt = build_prompt(q, chosen)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **ids, max_new_tokens=256, do_sample=True, temperature=0.3, top_p=0.9, eos_token_id=tok.eos_token_id
        )
        out = tok.decode(gen[0], skip_special_tokens=True)
        if "[ASSISTANT]" in out:
            out = out.split("[ASSISTANT]")[-1].strip()

        sources = []
        for rank,(i,_) in enumerate(chosen,1):
            src = meta[i].get("source","")
            txt = meta[i].get("text","").replace("\n"," ")
            sources.append(f"[{rank}] {src}: {txt[:200]}...")
        return out, "\n\n".join(sources)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔎 RAG QA (bge-m3 + CrossEncoder + Qwen 7B)\n显示引用，尽量避免幻觉。")
        inp = gr.Textbox(label="Question", placeholder="Ask anything…")
        btn = gr.Button("Ask")
        out = gr.Textbox(label="Answer", lines=8)
        cites = gr.Textbox(label="Sources", lines=10)
        btn.click(fn=answer, inputs=inp, outputs=[out, cites])
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=False)

if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","1")
    main()
