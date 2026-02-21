import json, os
from pathlib import Path
def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False)+"\n")
def normalize_context(ctx_item):
    title, text = "", ""
    if isinstance(ctx_item,dict):
        title = ctx_item.get("title", "")
        sents = ctx_item.get("sentences", [])
        if isinstance(sents, (tuple,list)):
            text = "".join(map(str,sents))
        else:
            text = str(sents)
    elif isinstance(ctx_item, (list, tuple)):
        if(len(ctx_item))>=2:
            title = str(ctx_item[0])
            sents = ctx_item[1]
            if isinstance(sents,(list, tuple)):
                text = "\n".join(map(str,sents))
            else:
                text = str(sents)
    else:
        text = str(ctx_item)
    return title, text.strip()
def main():
    input_path = "data/raw/hotpot_train_v1.1.json"
    with open(input_path, 'r', encoding = 'utf-8') as f:
        hp_data = json.load(f)
    hp_subset = hp_data[:20000]
    corpus, qa = [], []
    for data in hp_subset:
        for ctx_item in data["context"]:
            title, text = normalize_context(ctx_item)
            if text:
                corpus.append({"source": f"{title}.hotpot" if title else "hotpot","text" : text})
        qa.append({"question": data["question"], "answer": data["answer"]})
    dump_jsonl("data/processed/corpus.jsonl", corpus)
    dump_jsonl("data/processed/hp_eval_eq.jsonl", qa)
    print(f"共计{len(corpus)}条语料， {len(qa)}条评测问答")
if __name__ == "__main__":
    main()
            
                
                


    
        

    