import json, os
from pathlib import Path
from datasets import load_dataset

def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_context(ctx_item):
    title, text = "", ""
    if isinstance(ctx_item, dict):
        title = ctx_item.get("title", "")
        sents = ctx_item.get("sentences", [])
        if isinstance(sents, (list, tuple)):
            text = "\n".join(map(str, sents))
        else:
            text = str(sents)
    elif isinstance(ctx_item, (list, tuple)):
        if len(ctx_item) >= 2:
            title = str(ctx_item[0])
            sents = ctx_item[1]
            if isinstance(sents, (list, tuple)):
                text = "\n".join(map(str, sents))
            else:
                text = str(sents)
        else:
            text = " ".join(map(str, ctx_item))
    else:
        text = str(ctx_item)
    return title, text.strip()

def main():
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    hp = load_dataset("hotpot_qa", "fullwiki", split="validation[:200]")

    corpus, qa = [], []
    for ex in hp:
        for ctx_item in ex["context"]:
            title, text = normalize_context(ctx_item)
            if text:
                corpus.append({"source": f"{title}.hotpot" if title else "hotpot", "text": text})
        qa.append({"question": ex["question"], "answer": ex["answer"]})

    dump_jsonl("data/raw/hp_corpus.jsonl", corpus)
    dump_jsonl("data/raw/hp_eval_qa.jsonl", qa)
    print("OK corpus:", len(corpus), "| qa:", len(qa))

if __name__ == "__main__":
    main()
