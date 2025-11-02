import argparse, json, os, random, re

def read_jsonl(p):
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def clean_text(s:str)->str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default="data/raw/hp_corpus.jsonl")
    ap.add_argument("--qa", default="data/raw/hp_eval_qa.jsonl")
    ap.add_argument("--out_corpus", default="data/clean/hp_corpus.clean.jsonl")
    ap.add_argument("--out_qa", default="data/clean/hp_eval_qa.clean.jsonl")
    args = ap.parse_args()

    # 1) 清洗语料（尽量不改 schema，只是去掉多余空白；保留原字段）
    seen = set()
    cleaned_corpus = []
    for r in read_jsonl(args.corpus):
        if "text" in r and isinstance(r["text"], str):
            r["text"] = clean_text(r["text"])
            if len(r["text"]) < 5: 
                continue
            sig = r["text"]
        else:
            # 没有 text 字段就跳过（防止后续索引失败）
            continue
        if sig in seen: 
            continue
        seen.add(sig)
        cleaned_corpus.append(r)
    write_jsonl(args.out_corpus, cleaned_corpus)

    # 2) QA 子集（不改字段名，只做抽样）
    qa_all = list(read_jsonl(args.qa))
    random.seed(args.seed)
    if args.n and args.n < len(qa_all):
        qa_all = random.sample(qa_all, args.n)
    write_jsonl(args.out_qa, qa_all)

    print(f"clean corpus -> {args.out_corpus} ({len(cleaned_corpus)})")
    print(f"subsample qa -> {args.out_qa} ({len(qa_all)})")

if __name__ == "__main__":
    main()
