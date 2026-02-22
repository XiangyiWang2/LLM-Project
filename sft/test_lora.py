import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
MODEL_ID = "Qwen/Qwen1.5-7B-Chat"
LORA_DIR = "checkpoints/qwen_lora"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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

    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    

  
    instruction = "When did Virgin Australia start operating?"
    context = "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route."


    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"


    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    

    outputs = model.generate(
        **inputs, 
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
    

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = response_text.split("### Response:\n")[-1]

    print("\n🤖 【Qwen-LoRA 的回答】:")
    print(final_answer)

if __name__ == "__main__":
    main()
