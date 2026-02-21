import os
import json
import faiss
import numpy as np
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
INDEX_DIR = "data/index"
EMBED_MODEL_NAME = "BAAI/bge-m3"
LLM_ID = "Qwen/Qwen1.5-7B-Chat"
LORA_DIR = "checkpoints/qwen_lora"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "meta.json"), 'r', encoding='utf-8') as f:
    documents = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", quantization_config=quantization_config)
llm = PeftModel.from_pretrained(base_model, LORA_DIR)
def rag_inference(question, top_k, temperature):
    q_emb = np.array(embed_model.encode([question], normalize_embeddings=True)).astype('float32')
    D, I = index.search(q_emb, top_k) 
    retrieved_content = ""
    source_list = []
    for i, idx in enumerate(I[0]):
        doc_text = documents[idx]['text'] if isinstance(documents[idx], dict) else documents[idx]
        source_list.append(f"【参考资料 {i+1}】\n{doc_text}")
        retrieved_content += doc_text + "\n---\n"
    prompt = f"### Instruction:\n{question}\n\n### Input:\n{retrieved_content}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_res.split("### Response:\n")[-1].strip()
    return answer, "\n\n".join(source_list)-
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Optimization Lab") as demo:
    gr.Markdown("# 🤖 RAG + SFT 垂直领域智能问答系统")
    gr.Markdown("本项目由 Qwen-7B-Chat + LoRA 微调 + BGE-M3 检索增强驱动。")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 控制面板")
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="检索参考条数 (Top-K)")
            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="生成随机度 (Temperature)")
            btn = gr.Button("🚀 提交提问", variant="primary")    
        with gr.Column(scale=2):
            gr.Markdown("### 💬 对话窗口")
            question_input = gr.Textbox(label="输入您的问题", placeholder="例如：苹果是什么？", lines=2)
            answer_output = gr.Textbox(label="Qwen 的回答", interactive=False, lines=10)        
        with gr.Column(scale=2):
            gr.Markdown("### 📚 检索证据链")
            sources_output = gr.Textbox(label="检索到的参考资料原文", interactive=False, lines=15)
    btn.click(
        fn=rag_inference, 
        inputs=[question_input, top_k_slider, temp_slider], 
        outputs=[answer_output, sources_output]
    )
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")