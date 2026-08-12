"""检测信号消融：单模型重采样 / 异构模型 / 完整方法 vs 随机基线
同一批 HaluEval 200 任务（100 样本 × right/hallucinated），只重跑检测部分（不 judge/reduce）
- score1 = 1 - consistent(a, a1)：仅主模型高温重答（pro, T=1.0）
- score2 = 1 - consistent(a, a2)：仅异构模型回答（flash, T=0.7）
- score  = (score1+score2)/2：完整方法
各算 AUROC / PR-AUC / AURC + bootstrap 95% CI
用法: python3 ablation_detect.py [--samples 100]
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from halu_loop import load_halu, consistent, gen, C, DS, MAIN, aurc, SAMP, CONSIST

RNG = np.random.default_rng(42)

def detect_split(it, answer):
    a1 = gen(MAIN, 1.0, SAMP.format(k=it["knowledge"], q=it["question"]))
    a2 = gen(DS, 0.7, SAMP.format(k=it["knowledge"], q=it["question"]))
    c1 = 1 if consistent(answer, a1) else 0
    c2 = 1 if consistent(answer, a2) else 0
    return {"c1": c1, "c2": c2,
            "score1": 1 - c1, "score2": 1 - c2, "score": 1 - (c1 + c2) / 2}

def boot_ci(scores, labels, n_boot=1000):
    n = len(labels)
    def stat(s, l):
        return roc_auc_score(l, s), average_precision_score(l, s), aurc(s, l)
    base = stat(scores, labels)
    boots = np.array([stat(scores[(idx := RNG.integers(0, n, n))], labels[idx]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return base, lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()
    raw = load_halu(args.samples)
    tasks = []
    for i, it in enumerate(raw):
        tasks.append({"it": it, "answer": it["right_answer"], "label": 0})
        tasks.append({"it": it, "answer": it["hallucinated_answer"], "label": 1})
    dets, fails = {}, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(detect_split, t["it"], t["answer"]): t for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                dets[(id(t["it"]), t["label"])] = f.result()
            except Exception:
                fails += 1
    tasks = [t for t in tasks if (id(t["it"]), t["label"]) in dets]
    print(f"有效 {len(tasks)} 任务（跳过 {fails}），耗时 {time.time()-t0:.0f}s")

    labels = np.array([t["label"] for t in tasks])
    score1 = np.array([dets[(id(t["it"]), t["label"])]["score1"] for t in tasks])
    score2 = np.array([dets[(id(t["it"]), t["label"])]["score2"] for t in tasks])
    score = np.array([dets[(id(t["it"]), t["label"])]["score"] for t in tasks])
    rnd = RNG.random(len(labels))

    print(f"\n{'信号':<22}{'AUROC':>10}{'PR-AUC':>10}{'AURC':>10}")
    rows = []
    for nm, s in [("随机基线", rnd), ("单模型重采样(pro T=1.0)", score1),
                  ("异构模型(flash T=0.7)", score2), ("完整方法(两者组合)", score)]:
        base, lo, hi = boot_ci(s, labels)
        rows.append((nm, base, lo, hi))
        print(f"{nm:<22}{base[0]:>10.3f}{base[1]:>10.3f}{base[2]:>10.3f}")
        print(f"{'':22}  CI: AUROC [{lo[0]:.3f},{hi[0]:.3f}]  PR-AUC [{lo[1]:.3f},{hi[1]:.3f}]  AURC [{lo[2]:.3f},{hi[2]:.3f}]")

    out = {"main": MAIN, "ds": DS, "n_tasks": len(tasks), "samples": args.samples,
           "results": [{"i": i, "label": int(labels[i]),
                        "score1": float(score1[i]), "score2": float(score2[i]), "score": float(score[i])}
                       for i in range(len(tasks))]}
    p = Path(__file__).parent / "results" / "ablation_detect.jsonl"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已保存 {p}")

if __name__ == "__main__":
    main()
