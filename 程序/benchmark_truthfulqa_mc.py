"""TruthfulQA 选择题（MC1）评测：单模型基线 + Self-Consistency
指标：准确率（MC1 = 单次回答 / Self-Consistency N 次采样多数投票）
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import get_client
from truthfulqa_common import (build_mc_prompt, extract_choice, load_truthfulqa_mc,
                               load_cache, save_cache)

RESULT_DIR = Path(__file__).parent / "results"


def eval_single(item, model, temperature, max_tokens=512):
    """单题单次回答"""
    q = item["question"]
    choices = item["mc1_choices"]
    correct_label = chr(65 + item["mc1_labels"].index(1))
    prompt = build_mc_prompt(q, choices)
    c = get_client(model, temperature=temperature, max_tokens=max_tokens)
    resp = c.generate([{"role": "user", "content": prompt}])
    pred = extract_choice(resp)
    return {
        "idx": item["idx"],
        "correct": correct_label,
        "pred": pred,
        "is_correct": pred == correct_label,
        "raw": resp,
    }


def run_single(model, max_samples=200, num_workers=10, cache_file="mc1_single.jsonl"):
    """基线：单次回答准确率"""
    items = load_truthfulqa_mc()[:max_samples]
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(eval_single, it, model, 0.0): it for it in items}
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: r["idx"])
    acc = sum(r["is_correct"] for r in results) / len(results)
    print(f"[{model}] MC1 单次回答: {acc*100:.2f}% ({sum(r['is_correct'] for r in results)}/{len(results)})")
    save_cache(RESULT_DIR / cache_file, {"model": model, "results": results})
    return results


def run_self_consistency(model, n=5, max_samples=200, num_workers=6, temperature=1.0, cache_file="mc1_sc.jsonl"):
    """Self-Consistency：N 次采样 → 多数投票
    返回每个样本的投票结果 + 各选项票数分布（后者也是幻觉检测的置信度信号）
    """
    items = load_truthfulqa_mc()[:max_samples]

    def eval_consistency(item):
        q = item["question"]
        choices = item["mc1_choices"]
        correct_label = chr(65 + item["mc1_labels"].index(1))
        prompt = build_mc_prompt(q, choices)
        c = get_client(model, temperature=temperature, max_tokens=512)
        votes = []
        for _ in range(n):
            resp = c.generate([{"role": "user", "content": prompt}])
            pred = extract_choice(resp)
            if pred:
                votes.append(pred)
        # 多数投票
        from collections import Counter
        cnt = Counter(votes)
        final = cnt.most_common(1)[0][0] if cnt else None
        return {
            "idx": item["idx"],
            "correct": correct_label,
            "final": final,
            "is_correct": final == correct_label,
            "vote_dist": dict(cnt),
            "n_votes": len(votes),
        }

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(eval_consistency, it): it for it in items}
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: r["idx"])
    acc = sum(r["is_correct"] for r in results) / len(results)
    print(f"[{model}] MC1 Self-Consistency N={n}: {acc*100:.2f}% ({sum(r['is_correct'] for r in results)}/{len(results)})")
    save_cache(RESULT_DIR / cache_file, {"model": model, "n": n, "temperature": temperature, "results": results})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ikun/gpt-5.6-sol")
    ap.add_argument("--n", type=int, default=5, help="Self-Consistency 采样次数")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--mode", default="both", choices=["single", "sc", "both"])
    args = ap.parse_args()

    t0 = time.time()
    if args.mode in ("single", "both"):
        run_single(args.model, args.max_samples)
    if args.mode in ("sc", "both"):
        run_self_consistency(args.model, n=args.n, max_samples=args.max_samples)
    print(f"总耗时: {time.time()-t0:.0f}s")
