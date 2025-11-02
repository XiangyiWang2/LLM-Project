import os, json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[warn] missing {p}")
        return None

# ---- 读 RAG 评测（Base / LoRA）----
base = load_json("reports/rag_eval_base.json")
lora = load_json("reports/rag_eval_lora.json")

metrics = [("Recall@k","Recall@k"), ("RerankHit@r","RerankHit@r"), ("AnswerHit","AnswerHit")]
labels  = [lab for _, lab in metrics]
base_vals = [ (base or {}).get(k, 0.0) for k,_ in metrics ]
lora_vals = [ (lora or {}).get(k, 0.0) for k,_ in metrics ]

x = np.arange(len(labels)); w = 0.35
fig = plt.figure(figsize=(7,4))
bars1 = plt.bar(x - w/2, base_vals, w, label="Base")
bars2 = plt.bar(x + w/2, lora_vals, w, label="LoRA")
for b in list(bars1)+list(bars2):
    h = float(b.get_height())
    plt.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("RAG Evaluation: Base vs LoRA")
plt.ylim(0.0, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("figures/metrics_bar.png", dpi=200)
plt.close(fig)
print("wrote figures/metrics_bar.png")

# ---- 读训练曲线（trainer_state）----
st = load_json("reports/sft_trainer_state.json") or load_json("ckpts/qwen7b_lora_dolly/trainer_state.json")
if st and "log_history" in st:
    logs = st["log_history"]

    sl = [(e["step"], e["loss"]) for e in logs if "step" in e and "loss" in e]
    if sl:
        steps, losses = zip(*sl)
        fig = plt.figure(figsize=(7,4))
        plt.plot(steps, losses)
        plt.xlabel("Step"); plt.ylabel("Training Loss"); plt.title("SFT Training Loss")
        plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
        plt.savefig("figures/sft_loss.png", dpi=200)
        plt.close(fig)
        print("wrote figures/sft_loss.png")

    s_lr = [(e["step"], e["learning_rate"]) for e in logs if "step" in e and "learning_rate" in e]
    if s_lr:
        lr_steps, lrs = zip(*s_lr)
        fig = plt.figure(figsize=(7,4))
        plt.plot(lr_steps, lrs)
        plt.xlabel("Step"); plt.ylabel("Learning Rate"); plt.title("SFT Learning Rate (schedule)")
        plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
        plt.savefig("figures/sft_lr.png", dpi=200)
        plt.close(fig)
        print("wrote figures/sft_lr.png")
else:
    print("[warn] no trainer_state found; skip curves")
