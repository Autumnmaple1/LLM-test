"""HLE 裸基准：纯文本题分层抽样，生成答案 + 记录 token usage
用法: python3 hle_bench.py --max-n 300 --model deepseek/deepseek-v4-flash"""
import argparse, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from llm_client import LLMClient

DATA = Path(__file__).parent / "data_hle" / "hle_test.parquet"


def load_text():
    df = pd.read_parquet(DATA)
    return df[df["image"].isna() | (df["image"] == "")].reset_index(drop=True)


def sample(df, max_n):
    if len(df) <= max_n:
        return df
    per = max(1, max_n // df["category"].nunique())
    idx = df.groupby("category", group_keys=False).head(per).index
    return df.loc[idx].reset_index(drop=True).head(max_n)


def prompt_for(row):
    if row["answer_type"] == "multipleChoice":
        return f"{row['question']}\n\nOutput only the letter of the correct answer."
    return f"{row['question']}\n\nAnswer concisely and precisely."


def answer_letter(text):
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else None


def run(max_n, model, workers=10):
    df = sample(load_text(), max_n)
    c = LLMClient(model)
    cache_p = Path(__file__).parent / "results" / f"hle_{model.split('/')[-1]}_{max_n}.jsonl"
    done = set()
    if cache_p.exists():
        for l in cache_p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                o = json.loads(l)
                if o.get("pred"):
                    done.add(o["id"])

    def one(row):
        qid = row["id"]
        if qid in done:
            return None
        resp = c.client.chat.completions.create(
            model=c.model, messages=[{"role": "user", "content": prompt_for(row)}],
            temperature=0.0, max_tokens=40000)
        content = resp.choices[0].message.content or ""
        u = resp.usage
        pred = answer_letter(content) if row["answer_type"] == "multipleChoice" else content.strip()
        rec = {"id": qid, "type": row["answer_type"], "category": row["category"],
               "question": row["question"], "answer": str(row["answer"]), "pred": pred,
               "ptok": u.prompt_tokens if u else 0, "ctok": u.completion_tokens if u else 0,
               "rtok": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) if u else 0}
        with open(cache_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    rows = list(df.iterrows())
    n = len(rows)
    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, r): i for i, (_, r) in enumerate(rows)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  单题失败: {type(e).__name__}: {str(e)[:80]}", flush=True)
            n_done += 1
            if n_done % 25 == 0 or n_done == n:
                print(f"  进度 {n_done}/{n} ({n_done/n*100:.0f}%)", flush=True)
    print(f"完成 {n} 题", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=300)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    args = ap.parse_args()
    t0 = time.time()
    run(args.max_n, args.model)
    print(f"耗时 {time.time()-t0:.0f}s")
