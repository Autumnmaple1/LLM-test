"""FActScore 对比：对 closed_loop 结果中触发修正的样本，重跑流程拿原文，算主回答 vs 最终回答的事实准确率"""
import argparse, json
from pathlib import Path
from factscore import decompose_claims, verify_claims
from llm_client import get_client
from truthfulqa_common import load_truthfulqa_gen
import closed_loop as cl

RESULT_DIR = Path(__file__).parent / "results"
PRO = "deepseek/deepseek-v4-pro"
FLASH = "deepseek/deepseek-v4-flash"

def factscore(c, jc, answer, reference):
    claims = decompose_claims(c, answer)
    verifs = verify_claims(jc, claims, reference)
    if not verifs:
        return 0.0, 0
    supported = sum(1 for v, _ in verifs if v == "SUPPORT")
    return supported / len(verifs), len(verifs)

def main(cache_file, max_n=30):
    d = json.loads((RESULT_DIR / cache_file).read_text(encoding="utf-8"))
    refs = {it["idx"]: it["best_answer"] for it in load_truthfulqa_gen()}
    items = [r for r in d["results"] if r["triggered"]][:max_n]
    if not items:
        print("没有触发修正的样本")
        return
    c = get_client(PRO, temperature=0.0)
    jc = get_client(FLASH, temperature=0.0)
    print(f"对 {len(items)} 条触发修正的样本计算 FActScore...")
    for r in items:
        idx = r["idx"]
        fs_o, n_o = factscore(c, jc, r["main"], refs[idx])
        fs_n, n_n = factscore(c, jc, r["final"], refs[idx])
        r["fs_main"] = fs_o
        r["fs_final"] = fs_n
        r["fs_claims"] = (n_o, n_n)
        print(f"  idx={idx} FActScore {fs_o:.2f} -> {fs_n:.2f} (claims {n_o}->{n_n})")
    avg_o = sum(r["fs_main"] for r in items) / len(items)
    avg_n = sum(r["fs_final"] for r in items) / len(items)
    up = sum(1 for r in items if r["fs_final"] > r["fs_main"])
    print(f"\nFActScore 均值: {avg_o*100:.1f}% -> {avg_n*100:.1f}% | 提升 {up}/{len(items)} 条")
    d["factscore"] = {"n": len(items), "main": avg_o, "final": avg_n, "improved": up / len(items)}
    (RESULT_DIR / cache_file).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="closed_loop_t0.67.jsonl")
    ap.add_argument("--max-n", type=int, default=30)
    args = ap.parse_args()
    main(args.cache, args.max_n)
