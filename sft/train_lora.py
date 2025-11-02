import argparse, json, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          BitsAndBytesConfig)
from trl import SFTTrainer
from peft import LoraConfig

def load_jsonl(p):
    return [json.loads(l) for l in open(p,'r',encoding='utf-8') if l.strip()]

def to_text(ex):
    # 通用指令模板（Qwen/Llama/Mistral 都兼容）
    sys = "You are a helpful assistant."
    instr = ex.get("instruction","").strip()
    inp   = ex.get("input","").strip()
    out   = ex.get("output","").strip()
    user = instr if not inp else f"{instr}\n\nInput:\n{inp}"
    return (f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n"
            f"[USER]\n{user}\n[/USER]\n[ASSISTANT]\n{out}</s>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--data',  default='data/raw/dolly15k.jsonl')
    ap.add_argument('--out',   default='ckpts/qwen7b_lora_dolly')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--seq',    type=int, default=1024)
    ap.add_argument('--bsz',    type=int, default=1)
    ap.add_argument('--ga',     type=int, default=32)
    ap.add_argument('--lr',     type=float, default=1e-4)
    ap.add_argument('--r',      type=int, default=16)
    ap.add_argument('--alpha',  type=int, default=32)
    ap.add_argument('--dropout',type=float, default=0.05)
    ap.add_argument('--qlora',  action='store_true')
    args = ap.parse_args()

    print("loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant
    )
    model.config.use_cache = False

    print("preparing dataset...")
    rows = load_jsonl(args.data)
    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "up_proj","down_proj","gate_proj"]  # 适配 Qwen/Llama/Mistral
    )

    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_32bit" if args.qlora else "adamw_torch",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        peft_config=lora,
        dataset_text_field="text",
        max_seq_length=args.seq,
        packing=False,
        args=train_args
    )

    print("start training...")
    trainer.train()
    print("saving adapter...")
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("done ->", args.out)

if __name__ == "__main__":
    main()
