"""HLE 判分 + token 汇总：multipleChoice 字母匹配，exactMatch LLM judge
用法: python3 hle_judge.py --cache hle_deepseek-v4-flash_300.jsonl --judge deepseek/deepseek-v4-flash"""
import argparse, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import LLMClient

RES = Path(__file__).parent / "results"


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9=<>+\-.,/() ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,")


def judge_one(c, row):
    q, a, p = row["question"], row["answer"], row["pred"]
    if not p:
        return {"id": row["id"], "judge": "NO", "norm_match": False}
    exact = norm(p) == norm(a) or norm(a) in norm(p) or norm(p) in norm(a)
    if exact:
        return {"id": row["id"], "judge": "YES", "norm_match": True}
    resp = c.generate([{"role": "user", "content":
        f"Question: {q}\nReference answer: {a}\nModel answer: {p}\n"
        "Is the model answer correct (equivalent to the reference answer)? Answer only YES or NO."}],
        temperature=0.0, max_tokens=2000, thinking=False)
    return {"id": row["id"], "judge": "YES" if resp.strip().upper().startswith("YES") else "NO",
            "norm_match": False}


def main(cache, judge_model, workers=10):
    raw = [json.loads(l) for l in (RES / cache).read_text(encoding="utf-8").splitlines() if l.strip()]
    # 去重：同 id 保留最后一条有 pred 的行；空 pred 行占位（未答按错计）
    best = {}
    for r in raw:
        if r.get("pred"):
            best[r["id"]] = r
        else:
            best.setdefault(r["id"], r)
    rows = list(best.values())
    mc = [r for r in rows if r["type"] == "multipleChoice"]
    em = [r for r in rows if r["type"] == "exactMatch"]
    mc_acc = sum(1 for r in mc if (r["pred"] or "").strip() == str(r["answer"]).strip())
    print(f"multipleChoice: {mc_acc}/{len(mc)} = {mc_acc/max(len(mc),1)*100:.1f}%")

    c = LLMClient(judge_model)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_one, c, r): r for r in em}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  judge失败: {type(e).__name__}: {str(e)[:80]}", flush=True)
    jmap = {r["id"]: r["judge"] for r in results}
    em_acc = sum(1 for r in em if jmap.get(r["id"]) == "YES")
    print(f"exactMatch: {em_acc}/{len(em)} = {em_acc/max(len(em),1)*100:.1f}%")

    total = len(rows)
    all_acc = mc_acc + em_acc
    print(f"总体: {all_acc}/{total} = {all_acc/total*100:.1f}%")

    pt = sum(r["ptok"] for r in rows)
    ct = sum(r["ctok"] for r in rows)
    rt = sum(r["rtok"] for r in rows)
    print(f"token: prompt={pt} completion={ct} reasoning={rt} total={pt+ct}")
    print(f"平均/题: prompt={pt/total:.0f} completion={ct/total:.0f} reasoning={rt/total:.0f}")
    with open(RES / f"hle_judge_{cache}.json", "w", encoding="utf-8") as f:
        json.dump({"mc": mc_acc, "mc_n": len(mc), "em": em_acc, "em_n": len(em),
                   "acc": all_acc / total, "tokens": {"prompt": pt, "completion": ct,
                   "reasoning": rt, "total": pt + ct}}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge", default="deepseek/deepseek-v4-flash")
    args = ap.parse_args()
    main(args.cache, args.judge)
