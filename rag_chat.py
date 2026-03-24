import os
import json
import faiss
import numpy as np
import torch
# [修改1] 增加导入 CrossEncoder
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

INDEX_DIR = "data/index"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-large"
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
LORA_DIR = "checkpoints/qwen_lora"

def main():
    print("Downloading BGE-M3")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Downloading BGE-Reranker")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Downloading FAISS")
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.json"), 'r', encoding='utf-8') as f:
        documents = json.load(f)
        
    print("Downloading 4-bit ground model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    print("LoRA weights")
    llm = PeftModel.from_pretrained(base_model, LORA_DIR)

    while True:
        query = input("\n Question:(exit with q) ")
        if query.lower() == 'q':
            break
        if not query.strip():
            continue
        q_embedding = embedding_model.encode([query], normalize_embeddings=True)
        q_embedding = np.array(q_embedding).astype('float32')
        D, I = index.search(q_embedding, 10)
        
        initial_texts = []
        for idx in I[0]:
            document = documents[idx]
            initial_texts.append(document['text'])
        cross_inp = [[query, text] for text in initial_texts]
        rerank_scores = reranker.predict(cross_inp)        
        scored_texts = list(zip(rerank_scores, initial_texts))
        scored_texts.sort(key=lambda x: x[0], reverse=True)  
        top_3_texts = [text for score, text in scored_texts[:3]]
        litm_texts = [top_3_texts[0], top_3_texts[2], top_3_texts[1]]
        merge_text = "\n---\n".join(litm_texts)

        prompt = f"### Instruction:\n{query}\n\n### Input:\n{merge_text}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors='pt').to("cuda")        
        output = llm.generate(
            **inputs,
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response_text.split("### Response:\n")[-1].strip()        
        print("\nQwen:")
        print(answer)
        print("-" * 50)

if __name__ == '__main__':
    main()
