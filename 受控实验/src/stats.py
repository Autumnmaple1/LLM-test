"""统计分析：从 results/raw 生成汇总（正确率/检测/策略对照/成本/CI）
用法: python3 src/stats.py [--dataset all|mmlupro|hle|truthfulqa]
输出: results/summary/{dataset}_summary.md
统计规范：按题为单位；Wilson CI；按题 bootstrap 1000 次；配对 bootstrap 显著性；随机触发×30 均值±sd
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from halu_loop import aurc

BASE = Path(__file__).parent.parent
RAW = BASE / "results" / "raw"
SUM = BASE / "results" / "summary"
SUM.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)
CAND_COST = 3  # 触发时额外候选成本（a1+a2+b1）

def load_rows(dataset, group, raw_dir=None):
    p = (raw_dir or RAW) / f"{dataset}_{group}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def wilson(p, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))

def boot_ci_pairs(scores, labels, pairs, n_boot=1000):
    """按题成对 bootstrap（pairs: 每题任务索引列表；默认每题为一行）"""
    n = len(labels)
    def stat(s, l):
        return roc_auc_score(l, s), average_precision_score(l, s), aurc(s, l)
    base = stat(scores, labels)
    idxs = list(range(n))
    def draw():
        if pairs:
            sel = RNG.integers(0, len(pairs), len(pairs))
            idx = np.concatenate([pairs[j] for j in sel])
        else:
            idx = RNG.integers(0, n, n)
        return scores[idx], labels[idx]
    boots = np.array([stat(*draw()) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return base, lo, hi

def summary_group(rows, tag, all_qids=None):
    if all_qids:
        qids = {r["qid"] for r in rows}
        for q in all_qids:
            if q not in qids:
                rows.append({"qid": q, "a0_correct": False, "final_correct": False, "calls": [], "skipped": True})
        rows.sort(key=lambda r: r["qid"])
    n = len(rows)
    a0c = np.array([r["a0_correct"] for r in rows])
    fc = np.array([r["final_correct"] for r in rows])
    lines = [f"### {tag}（n={n}）"]
    lines.append(f"- 原始正确率: {a0c.mean()*100:.1f}% [{wilson(a0c.mean(), n)[0]*100:.1f}, {wilson(a0c.mean(), n)[1]*100:.1f}]")
    lines.append(f"- 最终正确率: {fc.mean()*100:.1f}% [{wilson(fc.mean(), n)[0]*100:.1f}, {wilson(fc.mean(), n)[1]*100:.1f}]")
    lines.append(f"- 净提升: {(fc.mean()-a0c.mean())*100:+.1f}pt")
    t = Counter(zip(a0c, fc))
    lines.append(f"- 四格: 错→对 {t[(False, True)]}、错→错 {t[(False, False)]}、对→对 {t[(True, True)]}、对→错 {t[(True, False)]}")
    wr = t[(True, False)] / max(t[(True, True)] + t[(True, False)], 1)
    fr = t[(False, True)] / max(t[(False, True)] + t[(False, False)], 1)
    lines.append(f"- 误改率: {wr*100:.1f}%（正确被改错/正确总数） | 错误修正率: {fr*100:.1f}%")
    calls = [len(r["calls"]) for r in rows]
    cost = sum(c["cost"] for r in rows for c in r["calls"])
    lat = sorted(c["latency"] for r in rows for c in r["calls"])
    retries = sum(c["retries"] for r in rows for c in r["calls"])
    lat_s = f"{lat[len(lat)//2]:.0f}/{lat[int(len(lat)*0.95)]:.0f}" if lat else "-"
    lines.append(f"- 调用: {np.mean(calls):.1f}/题 | 费用: {cost:.3f} 元（{cost/max(n,1):.4f}/题） | 延迟 P50/P95: {lat_s}s | 重试: {retries} 次"
                 + ("（含跳过题按错误计）" if all_qids else ""))
    return "\n".join(lines), {"a0c": a0c, "fc": fc}

def detect_section(own, lines):
    """风险分数检测效果（label = a0 判分错误）"""
    rows = own
    if not rows:
        return
    scores = np.array([r["risk"] for r in rows])
    labels = np.array([not r["a0_correct"] for r in rows])
    base, lo, hi = boot_ci_pairs(scores, labels, None)
    lines.append("\n### 风险分数检测效果（label = 原始回答错误）")
    for nm, v, l, h in zip(["AUROC", "PR-AUC", "AURC"], base, lo, hi):
        lines.append(f"- {nm}: {v:.3f} [{l:.3f}, {h:.3f}]")
    for thr in [0.3, 0.5, 0.7]:
        k = max(1, int(thr * len(rows)))
        top = np.argsort(-scores)[:k]
        P = labels[top].mean()
        R = labels[top].sum() / max(labels.sum(), 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        lines.append(f"- 触发率 {thr*100:.0f}%: 触发样本错误率 {labels[top].mean()*100:.1f}%（全体 {labels.mean()*100:.1f}%） | Precision={P:.2f} Recall={R:.2f} F1={F1:.2f}")
    dist = Counter(scores)
    lines.append(f"- risk 分布: {dict(sorted(dist.items()))}")

def strategy_section(own, lines):
    """从 own 缓存（所有题真实修正过）构造策略对照"""
    rows = own
    if not rows:
        return
    n = len(rows)
    a0c = np.array([r["a0_correct"] for r in rows])
    fc = np.array([r["final_correct"] for r in rows])
    scores = np.array([r["risk"] for r in rows])
    lines.append("\n### 触发策略对照（修正均已真实执行，从 own 缓存构造）")
    lines.append("| 策略 | 正确率 | 调用/题 |")
    lines.append("|---|---|---|")
    base_acc = a0c.mean()
    lines.append(f"| 裸答（不触发） | {base_acc*100:.1f}% | 1.0 |")
    all_acc = fc.mean()
    lines.append(f"| 全量修正（所有题触发） | {all_acc*100:.1f}% | {1+CAND_COST:.1f} |")
    for thr in [0.3, 0.5, 0.7]:
        k = max(1, int(thr * n))
        sel = np.zeros(n, bool); sel[np.argsort(-scores)[:k]] = True
        sel_acc = np.where(sel, fc, a0c).mean()
        rand_accs = []
        for _ in range(30):
            m = RNG.random(n) < (k / n)
            rand_accs.append(np.where(m, fc, a0c).mean())
        lines.append(f"| 选择性触发 {thr*100:.0f}% | {sel_acc*100:.1f}% | {1+CAND_COST*(k/n):.2f} |")
        lines.append(f"| 随机触发 {thr*100:.0f}%（×30） | {np.mean(rand_accs)*100:.1f}±{np.std(rand_accs)*100:.1f}% | {1+CAND_COST*(k/n):.2f} |")
    oracle = np.where(~a0c, fc, a0c).mean()
    err_rate = (~a0c).mean()
    lines.append(f"| Oracle（只修正错误样本） | {oracle*100:.1f}% | {1+CAND_COST*err_rate:.2f} |")
    # 配对显著性：选择性 vs 随机（同触发率）
    k = max(1, int(0.5 * n))
    sel = np.zeros(n, bool); sel[np.argsort(-scores)[:k]] = True
    diffs = []
    for _ in range(1000):
        idx = RNG.integers(0, n, n)
        m = RNG.random(n) < 0.5
        d_sel = np.where(sel[idx], fc[idx], a0c[idx]).mean()
        d_rnd = np.where(m[idx], fc[idx], a0c[idx]).mean()
        diffs.append(d_sel - d_rnd)
    diffs = np.array(diffs)
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lines.append(f"\n选择性 vs 随机（触发率 50%，配对 bootstrap 1000 次）：Δ = {diffs.mean()*100:+.1f}pt [{np.percentile(diffs,2.5)*100:+.1f}, {np.percentile(diffs,97.5)*100:+.1f}]，p = {p:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--raw-dir", default=None, help="自定义 raw 目录（如备份的 pro 主版本数据）")
    ap.add_argument("--out-dir", default=None, help="汇总输出目录")
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW
    out_dir = Path(args.out_dir) if args.out_dir else SUM
    out_dir.mkdir(exist_ok=True)
    datasets = ["mmlupro", "hle", "truthfulqa"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        doc = [f"# {ds} 实验结果汇总（受控实验）\n"]
        own = load_rows(ds, "own", raw_dir)
        bare = load_rows(ds, "bare", raw_dir)
        all_qids = [r["qid"] for r in bare] if bare else None
        for g, tag in [("bare", "Bare（单次回答）"), ("sc", "Self-Consistency K=3"),
                       ("cove", "CoVe（自我核查）"), ("own", "本人方案（多角度重答+冲突检测+修正）")]:
            rows = load_rows(ds, g, raw_dir)
            if rows:
                txt, _ = summary_group(rows, tag, all_qids)
                doc.append(txt)
        for v, tag in [("low", "消融：同模型低温重答"), ("high", "消融：同模型高温重答"), ("cross", "消融：跨模型 A+B")]:
            rows = load_rows(ds, v, raw_dir)
            if rows:
                acc = np.mean([r["a0_correct"] for r in rows])
                lines = [f"### {tag}（n={len(rows)}）", f"- 原始正确率: {acc*100:.1f}%"]
                # 该候选来源的 risk 检测力（label = a0 错误）
                s = np.array([r["risk"] for r in rows]); l = np.array([not r["a0_correct"] for r in rows])
                if len(set(s)) > 1 and l.sum() > 0 and l.sum() < len(l):
                    b, lo, hi = boot_ci_pairs(s, l, None)
                    lines.append(f"- risk 检测力（label=回答错误）: AUROC {b[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}] | PR-AUC {b[1]:.3f} | AURC {b[2]:.3f}")
                doc.append("\n".join(lines))
        if own:
            detect_section(own, doc)
            strategy_section(own, doc)
        md = "\n".join(doc)
        (out_dir / f"{ds}_summary.md").write_text(md, encoding="utf-8")
        print(f"[{ds}] 汇总已写入 {out_dir / f'{ds}_summary.md'}")

if __name__ == "__main__":
    main()
