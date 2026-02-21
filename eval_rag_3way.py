import json
import os
import re
import string
import faiss
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
INDEX_DIR = "data/index"
QA_FILE = "data/processed/hp_eval_eq.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-m3"
LLM_ID = "Qwen/Qwen1.5-7B-Chat"
LORA_DIR = "checkpoints/qwen_lora"
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.json"), 'r', encoding='utf-8') as f:
        documents = json.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", quantization_config=quantization_config)
    llm = PeftModel.from_pretrained(base_model, LORA_DIR)
    import random

    eval_data = []
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        for line in f: eval_data.append(json.loads(line))
    random.seed(42)
    eval_data = random.sample(eval_data, min(200, len(eval_data)))
            

    print(f"Starting evaluate {len(eval_data)} data...")
    metrics = {
        "base_no_rag": {"em": 0, "f1_sum": 0.0},
        "ft_no_rag": {"em": 0, "f1_sum": 0.0},
        "ft_with_rag": {"em": 0, "f1_sum": 0.0}
    }

    for data in tqdm(eval_data, desc="Evaluating (Base vs FT vs FT+RAG)"):
        question = data['question']
        ground_truth = data['answer']
        prompt_no_rag = f"### Instruction:\n{question}\n\n### Input:\n\n\n### Response:\n"
        inputs_no = tokenizer(prompt_no_rag, return_tensors="pt").to("cuda")
        with llm.disable_adapter():
            out_base = llm.generate(**inputs_no, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            pred_base = tokenizer.decode(out_base[0], skip_special_tokens=True).split("### Response:\n")[-1].strip()
            metrics["base_no_rag"]["em"] += exact_match_score(pred_base, ground_truth)
            metrics["base_no_rag"]["f1_sum"] += f1_score(pred_base, ground_truth)

        out_ft_no = llm.generate(**inputs_no, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        pred_ft_no = tokenizer.decode(out_ft_no[0], skip_special_tokens=True).split("### Response:\n")[-1].strip()
        metrics["ft_no_rag"]["em"] += exact_match_score(pred_ft_no, ground_truth)
        metrics["ft_no_rag"]["f1_sum"] += f1_score(pred_ft_no, ground_truth)
        q_emb = np.array(embed_model.encode([question], normalize_embeddings=True)).astype('float32')
        D, I = index.search(q_emb, 3)
        retrieved_texts = [documents[idx]['text'] if isinstance(documents[idx], dict) else documents[idx] for idx in I[0]]
        merged_context = "\n---\n".join(retrieved_texts)
        prompt_with_rag = f"### Instruction:\n{question}\n\n### Input:\n{merged_context}\n\n### Response:\n"
        inputs_with = tokenizer(prompt_with_rag, return_tensors="pt").to("cuda")
        out_ft_with = llm.generate(**inputs_with, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        pred_ft_with = tokenizer.decode(out_ft_with[0], skip_special_tokens=True).split("### Response:\n")[-1].strip()
        metrics["ft_with_rag"]["em"] += exact_match_score(pred_ft_with, ground_truth)
        metrics["ft_with_rag"]["f1_sum"] += f1_score(pred_ft_with, ground_truth)
    total = len(eval_data)
    results = {
        "base_em": metrics["base_no_rag"]["em"] / total * 100,
        "base_f1": metrics["base_no_rag"]["f1_sum"] / total * 100,
        "ft_no_em": metrics["ft_no_rag"]["em"] / total * 100,
        "ft_no_f1": metrics["ft_no_rag"]["f1_sum"] / total * 100,
        "ft_with_em": metrics["ft_with_rag"]["em"] / total * 100,
        "ft_with_f1": metrics["ft_with_rag"]["f1_sum"] / total * 100,
    }

    print("\n" + "="*60)
    print("📈 [Report: Base vs FT vs FT+RAG]")
    print(f"1. Original Model  -> EM: {results['base_em']:5.2f}% | F1: {results['base_f1']:5.2f}%")
    print(f"2. Qwen-LoRA  -> EM: {results['ft_no_em']:5.2f}% | F1: {results['ft_no_f1']:5.2f}%")
    print(f"3. Qwen-LoRA + RAG   -> EM: {results['ft_with_em']:5.2f}% | F1: {results['ft_with_f1']:5.2f}%")
    print("="*60)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = ['Exact Match (EM)', 'F1 Score']
    base_scores = [results['base_em'], results['base_f1']]
    ft_no_scores = [results['ft_no_em'], results['ft_no_f1']]
    ft_with_scores = [results['ft_with_em'], results['ft_with_f1']]

    x = np.arange(len(labels))
    width = 0.25 
    rects1 = ax.bar(x - width, base_scores, width, label='Base Qwen (Zero-shot)', color='#95A5A6') 
    rects2 = ax.bar(x, ft_no_scores, width, label='Qwen+LoRA (No RAG)', color='#E24A33')         
    rects3 = ax.bar(x + width, ft_with_scores, width, label='Qwen+LoRA + RAG', color='#348ABD')    

    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Ablation Study: Base vs LoRA vs RAG', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plot_path = "rag_3way_report.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Reseverd to: {plot_path}")

if __name__ == "__main__":
    main()