import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
INDEX_DIR = "data/index"
EMBEDDING_MODEL = "BAAI/bge-m3"
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
LORA_DIR = "checkpoints/qwen_lora"
def main():
    print("Downloading BGE-M3")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("Downloading FAISS")
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.json"), 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print("Downloading 4-bit ground model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code = True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map = "auto",
        quantization_config = quantization_config,
        trust_remote_code = True
    )
    print("LoRA weights")
    llm = PeftModel.from_pretrained(base_model, LORA_DIR)
    while True:
        query = input("\n Question:(exit with q)")
        if query.lower() == 'q':
            break
        if not query.strip():
            continue

        q_embedding = embedding_model.encode([query], normalize_embeddings=True)
        q_embedding = np.array(q_embedding).astype('float32')
        D, I = index.search(q_embedding, 3)
        retrieve_text = []
        for i, idx in enumerate(I[0]):
            document = documents[idx]
            text = document['text']
            retrieve_text.append(text)
            print(f"{i+1} similarity {D[0][i]:.4f}, text:{text[:200]}")
        merge_text = "\n---\n".join(retrieve_text)
        prompt = f"### Instruction:\n{query}\n\n### Input:\n{merge_text}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors = 'pt').to("cuda")
        output = llm.generate(
            **inputs,
            max_new_tokens = 150,
            pad_token_id = tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(output[0], skip_special_tokens = True)
        answer = response_text.split("### Response:\n")[-1].strip()
        print("Qwen:" )
        print(answer)
        print("-"*50)
if __name__ == '__main__':
    main()



