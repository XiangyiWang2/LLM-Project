import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 配置 ---
MODEL_ID = "Qwen/Qwen1.5-7B-Chat"
LORA_DIR = "checkpoints/qwen_lora"

def main():
    print("🚀 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("🧠 正在加载 4-bit 底座模型...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    print("✨ 正在注入你训练的 LoRA 灵魂...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    
    print("\n✅ 加载完成！开始测试...")

    # 我们拿你刚才那条 Virgin Australia 的数据来考考它
    instruction = "When did Virgin Australia start operating?"
    context = "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route."

    # 严格按照训练时的格式拼装 prompt
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"

    print("\n" + "="*40)
    print("【我们的提示词 (Prompt)】:")
    print(prompt)
    print("="*40)

    # 推理生成
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 让模型最多生成 50 个 token 的回答
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码并提取模型的纯净回答
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = response_text.split("### Response:\n")[-1]

    print("\n🤖 【Qwen-LoRA 的回答】:")
    print(final_answer)

if __name__ == "__main__":
    main()