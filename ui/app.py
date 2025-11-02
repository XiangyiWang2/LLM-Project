import argparse, torch, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_name, peft_path=None):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    if peft_path:
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    return tok, model

def generate_answer(user_text, system_text, max_new_tokens, do_sample, temperature):
    if not user_text.strip():
        return ""
    msgs = [
        {"role":"system","content":system_text or "You are a helpful assistant."},
        {"role":"user","content":user_text}
    ]
    inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # 简单切分拿回答
    return text.split("assistant")[-1].strip() if "assistant" in text else text.strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--peft_path", default=None)
    args = ap.parse_args()

    global tokenizer, model
    tokenizer, model = load_model(args.model, args.peft_path)

    with gr.Blocks(title="LLM-Project Demo") as demo:
        gr.Markdown("# LLM-Project Demo (Qwen2.5 + optional LoRA)")
        sys = gr.Textbox(label="System Prompt", value="You are a helpful assistant.")
        inp = gr.Textbox(label="User Input", lines=4, placeholder="Ask something…")
        with gr.Row():
            max_new = gr.Slider(32, 512, value=256, step=1, label="max_new_tokens")
            do_sample = gr.Checkbox(value=False, label="do_sample")
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
        btn = gr.Button("Generate")
        out = gr.Textbox(label="Answer", lines=12)
        btn.click(generate_answer, inputs=[inp, sys, max_new, do_sample, temp], outputs=out)
    demo.launch(server_name="0.0.0.0", server_port=7860)
