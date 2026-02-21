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

# --- 1. 配置参数 ---
MODEL_ID = "Qwen/Qwen1.5-7B-Chat"
DATA_PATH = "data/raw/dolly15k.jsonl"  # 👈 完美！直接用你本地的数据
OUTPUT_DIR = "checkpoints/qwen_lora"

# LoRA 核心参数
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

def format_dolly(sample):
    """将 Alpaca 格式的数据转换为 Qwen 认识的指令格式"""
    
    # 1. 拿指令
    instruction = sample.get('instruction', '')
    
    # 2. 拿参考资料 (你的数据里叫 'input')
    context = sample.get('input', '')
    
    # 3. 拿答案 (你的数据里叫 'output')
    response = sample.get('output', '')
    
    # 4. 拼装成 Qwen 喜欢的格式
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    
    return {"text": prompt + response + "<|im_end|>"}
def main():
    print(f"🚀 正在加载模型: {MODEL_ID}...")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 4-bit 量化配置 (极其省显存的关键)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # 3. 加载底座模型
    print("🧠 正在加载 4-bit 底座模型 (这可能需要几分钟)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    # 4. 配置并挂载 LoRA 适配器
    print("🛠️ 正在挂载 LoRA 适配器...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # 🌟 见证奇迹的时刻
    model.print_trainable_parameters() 

    # 5. 加载本地数据
    print(f"📚 正在加载本地数据集: {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 【实验阶段专用】：只取前 1000 条数据快速跑通闭环！
    print("⚡ 抽取前 1000 条数据进行快速验证...")
    dataset = dataset.select(range(1000)) 
    
    dataset = dataset.map(format_dolly)
    
    def process_func(example):
        tokenized = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("✂️ 正在 Tokenize 数据...")
    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

    # 6. 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2, # 如果显存 OOM，改回 1
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        remove_unused_columns=False
    )

    # 7. 启动 Trainer
    print("🔥 开始 SFT 微调...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    
    print(f"✅ 训练完成！LoRA 权重已保存至: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()