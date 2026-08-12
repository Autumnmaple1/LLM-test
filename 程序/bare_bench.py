"""裸基准评测：MMLU / MMLU-Pro 选择题准确率（无检测无修正）
用法: python3 bare_bench.py --bench mmlu --max-n 500 --shots 5
"""
import argparse, json, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from llm_client import get_client

DATA = Path(__file__).parent / "data_mmlu"
MODEL = "deepseek/deepseek-v4-pro"

def fmt_mmlu(q, choices, idx):
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"Answer the following multiple-choice question. Output only the letter of the correct answer.\nQuestion: {q}\nOptions:\n{body}\nAnswer:"

def fmt_mmlupro(q, options):
    letters = "ABCDEFGHIJ"
    body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    return f"Answer the following multiple-choice question. Output only the letter of the correct answer.\nQuestion: {q}\nOptions:\n{body}\nAnswer:"

def load(bench):
    if bench == "mmlu":
        df = pd.read_parquet(DATA / "mmlu" / "all" / "test-00000-of-00001.parquet")
        dev = pd.read_parquet(DATA / "mmlu" / "all" / "dev-00000-of-00001.parquet")
        return df, dev, "ABCD"
    elif bench == "mmlupro":
        df = pd.read_parquet(DATA / "mmlupro" / "data" / "test-00000-of-00001.parquet")
        return df, None, "ABCDEFGHIJ"
    raise ValueError(bench)

def sample(df, max_n):
    """分层抽样：MMLU 按科目均匀抽，保证代表性"""
    if max_n is None or len(df) <= max_n:
        return df
    col = "subject" if "subject" in df.columns else "category"
    per = max(1, max_n // df[col].nunique())
    idx = df.groupby(col, group_keys=False).head(per).index
    return df.loc[idx].reset_index(drop=True).head(max_n)

def shot_prefix(dev, subject, k, bench):
    if dev is None:
        return ""
    sel = dev[dev["subject"] == subject].head(k)
    parts = []
    for _, r in sel.iterrows():
        parts.append(fmt_mmlu(r["question"], r["choices"], r["answer"]))
        parts.append("ABCD"[r["answer"]])
    return "\n".join(parts) + "\n\n" if parts else ""

def answer_letter(text):
    m = re.search(r"\b([A-J])\b", text.upper())
    return m.group(1) if m else None

def run(bench, max_n, shots, workers=12, model=MODEL):
    df, dev, letters = load(bench)
    df = sample(df, max_n)
    c = get_client(model, temperature=0.0, max_tokens=40000)
    # 增量缓存：按 question_id 记录结果，重跑跳过已完成
    import json as _json, os
    cache_p = Path(__file__).parent / "results" / f"cache_{bench}_{model.split('/')[-1]}_{max_n}.jsonl"
    done = {}
    if cache_p.exists():
        for l in cache_p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                o = _json.loads(l)
                done[o["qid"]] = o["pred"]

    def one(row):
        qid = str(row.get("question_id", row.get("question", "")))
        if bench == "mmlu":
            pre = shot_prefix(dev, row["subject"], shots, bench)
            prompt = pre + fmt_mmlu(row["question"], row["choices"], row["answer"])
            gold = "ABCD"[row["answer"]]
        else:
            prompt = fmt_mmlupro(row["question"], row["options"])
            gold = str(row["answer"]).upper()
        if qid in done:
            return gold, done[qid], qid, True
        resp = c.generate([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=40000) or ""
        pred = answer_letter(resp)
        return gold, pred, qid, False

    rows = list(df.iterrows())
    n = len(rows)
    done_n = 0
    correct = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, r): i for i, (_, r) in enumerate(rows)}
        for f in as_completed(futs):
            try:
                gold, pred, qid, cached = f.result()
            except Exception:
                gold, pred, qid, cached = None, None, None, False  # 单题失败算错，不卡线程池
            done_n += 1
            correct += 1 if pred and pred == gold else 0
            if pred and qid is not None:
                with open(cache_p, "a", encoding="utf-8") as fp:
                    fp.write(_json.dumps({"qid": qid, "pred": pred}) + "\n")
            if done_n % 25 == 0 or done_n == n:
                print(f"  进度 {done_n}/{n} ({done_n/n*100:.0f}%) 当前准确率 {correct/done_n*100:.1f}%", flush=True)
    acc = correct / n
    print(f"[{bench}] shots={shots} n={n} 准确率 {acc*100:.1f}% ({correct}/{n})")
    return acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["mmlu", "mmlupro"])
    ap.add_argument("--max-n", type=int, default=500)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    t0 = time.time()
    acc = run(args.bench, args.max_n, args.shots, model=args.model)
    print(f"耗时 {time.time()-t0:.0f}s")
