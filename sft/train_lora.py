import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType


MODEL_ID = "Qwen/Qwen1.5-7B-Chat"
DATA_PATH = "data/raw/dolly15k.jsonl"  
OUTPUT_DIR = "checkpoints/qwen_lora"


LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

def format_dolly(sample):

    instruction = sample.get('instruction', '')
    

    context = sample.get('input', '')
    

    response = sample.get('output', '')
    

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    
    return {"text": prompt + response + "<|im_end|>"}
def main():
    print(f"Loading: {MODEL_ID}...")
    

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )


    print("Loading 4-bit Model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )


    print("Uploading LoRA Matrix")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters() 


    print(f"loading {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    

    dataset = dataset.select(range(1000)) 
    
    dataset = dataset.map(format_dolly)
    
    def process_func(example):
        tokenized = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenize Dataset")
    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    
    print(f"Finished, Save to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
