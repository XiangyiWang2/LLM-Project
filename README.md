# LLM-Project — LoRA SFT + RAG (Qwen2.5-7B-Instruct)

**中文摘要**：用 Qwen2.5-7B-Instruct 做 LoRA 微调（Dolly15k），并在 HotpotQA 风格语料上构建简易 RAG（BGE 检索 + CrossEncoder 重排 + LLM 生成）。提供训练脚本、RAG 评测脚本与图表。  
**EN**: LoRA fine-tuning Qwen2.5-7B-Instruct on Dolly15k plus a simple RAG pipeline (BGE retriever, CrossEncoder reranker, LLM generation). Scripts & figures included.

## Structure
data/
raw/     # corpora & eval sets (Git LFS)
index/   # FAISS index (Git LFS)
figures/   # plots (loss/lr, RAG metrics)
reports/   # JSON metrics (RAG, trainer_state, etc.)
rag/       # retrieval / rerank / evaluation
sft/       # LoRA SFT training
tools/     # tiny utilities (plotting, etc.)
ui/        # minimal Gradio demo (app.py)
## Environment
```bash
pip install transformers peft trl accelerate sentence-transformers faiss-cpu gradio
# PyTorch with CUDA per your platform
accelerate launch sft/train_lora.py \
  --model  Qwen/Qwen2.5-7B-Instruct \
  --data   data/raw/dolly15k.jsonl \
  --out    ckpts/qwen7b_lora_dolly \
  --epochs 2 --seq 768 --bsz 1 --ga 16 --lr 1e-4 \
  --r 16 --alpha 32 --dropout 0.05
Uses bf16 when available, disables use_cache, saves every 100 steps (already set in script).
CUDA_VISIBLE_DEVICES=0 python rag/eval_rag_lora.py \
  --n 50 --top_k 20 --rerank_top 5 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --peft_path ckpts/qwen7b_lora_dolly \
  --save_json reports/rag_eval_lora.json
Retriever: BAAI/bge-m3  ·  Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
<p align="center">
  < img src="figures/rag_base_vs_lora.png" width="520"><br/>
  <em>RAG metrics (Base vs LoRA)</em>
</p >
python ui/app.py --peft_path ckpts/qwen7b_lora_dolly
# then open http://localhost:7860
Notes
 • Large artifacts (data/raw, data/index, ckpts) tracked with Git LFS.
 • Remove any secrets before pushing (API keys, tokens).
 • Hardware: RTX 4090.
 • License: MIT
