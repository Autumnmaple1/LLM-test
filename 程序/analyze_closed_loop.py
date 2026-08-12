"""对照/统计可靠性分析 v2（2026-08-12）
模式1（真实对照，推荐）：--cache full_reduce_cache.jsonl —— 全部任务真实执行过方案A修正，
  从真实 final 构造 全量/随机/选择性/Oracle 策略（可进论文）
模式2（离线模拟，仅参考）：--cache halu_a_t0.5.jsonl --offline —— 未触发样本无真实修正，输出标"离线模拟"
改进：
- bootstrap 按题成对重采样（200 任务 = 100 题 × 2，任务 2i/2i+1 同题）
- 成本口径：检测 4 次调用/任务（a1+a2+2×一致判断），触发修正另 +3
- 显著性：选择性 vs 随机 用 paired bootstrap 差值 + p 值
用法: python3 analyze_closed_loop.py [--cache full_reduce_cache.jsonl|halu_a_t0.5.jsonl] [--offline]
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from halu_loop import aurc

HERE = Path(__file__).parent
RESULT = HERE / "results" if (HERE / "results").exists() else HERE.parent / "results"
OUT = RESULT / "06_对照与消融"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)
DET_CALLS = 4  # detect: a1 + a2 + consistent×2（halu_loop 里写 3 是错的）
RED_CALLS = 3  # 触发修正: fresh + consistent + (revise)

def boot_ci(scores, labels, n_boot=1000, pairs=None):
    """按题成对 bootstrap（pairs: 每题的任务索引列表，默认 200 任务独立）"""
    n = len(labels)
    def stat(s, l):
        return roc_auc_score(l, s), average_precision_score(l, s), aurc(s, l)
    base = stat(scores, labels)
    if pairs is None:
        def draw():
            idx = RNG.integers(0, n, n)
            return scores[idx], labels[idx]
    else:
        npairs = len(pairs)
        def draw():
            sel = RNG.integers(0, npairs, npairs)
            idx = np.concatenate([pairs[j] for j in sel])
            return scores[idx], labels[idx]
    boots = np.array([stat(*draw()) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return base, lo, hi

def binom_ci(p, n):
    z = 1.96
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)

def stratify(rs, mask, labels):
    mo = np.array([r["main_ok"] for r in rs]); fo = np.array([r["final_ok"] for r in rs])
    lbl = np.array(labels)
    out = np.where(mask, fo, mo)
    hl = lbl == 1
    fix = ((mo == 0) & (fo == 1) & mask & hl).sum() / max(((mo == 0) & mask & hl).sum(), 1)
    wr = ((mo == 1) & (fo == 0) & mask & ~hl).sum() / max(((mo == 1) & ~hl).sum(), 1)
    return out.mean(), fix, wr

def analyze(name, rs, labels, offline, pairs):
    n = len(rs)
    scores = np.array([r["score"] for r in rs])
    mo = np.array([r["main_ok"] for r in rs]); fo = np.array([r["final_ok"] for r in rs])
    lines = [f"\n## {name}（n={n}）\n"]
    tag = " ⚠️ 离线模拟（未触发样本无真实修正，仅参考）" if offline else ""
    lines.append(f"> 真实对照实验（完整修正缓存）" if not offline else f"> 反事实模拟{tag}")
    base, lo, hi = boot_ci(scores, labels, pairs=pairs)
    lines.append("### 检测质量（按题成对 bootstrap 1000 次 95% CI）")
    for nm, v, l, h in zip(["AUROC", "PR-AUC", "AURC"], base, lo, hi):
        lines.append(f"- {nm}: {v:.3f} [{l:.3f}, {h:.3f}]")
    lines.append("\n### 触发对照（5 种策略）")
    k = round(0.475 * n)
    sel_idx = np.argsort(-scores)[:k]
    sel = np.zeros(n, bool); sel[sel_idx] = True
    n_rand_sim = 200
    rand_masks = np.array([RNG.random(n) < 0.475 for _ in range(n_rand_sim)])
    rows = []
    for nm2, mask in [("裸答（不修正）", np.zeros(n, bool)), ("全量重生成", np.ones(n, bool)),
                      ("随机触发 47.5%", None), ("选择性触发 47.5%", sel), ("Oracle（只修真幻觉）", labels == 1)]:
        if nm2.startswith("随机"):
            accs = [np.where(m, fo, mo).mean() for m in rand_masks]
            acc, sd = np.mean(accs), np.std(accs)
            trig = 0.475
            fix = np.mean([stratify(rs, m, labels)[1] for m in rand_masks])
            wr = np.mean([stratify(rs, m, labels)[2] for m in rand_masks])
            acc_s = f"{acc*100:.1f}±{sd*100:.1f}"
        else:
            acc = np.where(mask, fo, mo).mean()
            _, fix, wr = stratify(rs, mask, labels)
            trig = mask.mean()
            acc_s = f"{acc*100:.1f}"
        calls = DET_CALLS + RED_CALLS * trig
        rows.append((nm2, acc_s, fix, wr, calls))
    lines.append("| 策略 | 诚实率 | 错误修正率 | 误改率 | 调用/任务 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]*100:.1f}% | {r[3]*100:.1f}% | {r[4]:.2f} |")
    # 选择性 vs 随机：paired bootstrap 差值 + p 值
    if not offline:
        diffs = []
        npairs = len(pairs)
        for _ in range(1000):
            sel_pts = np.concatenate([pairs[j] for j in RNG.integers(0, npairs, npairs)])
            acc_sel = np.where(sel[sel_pts], fo[sel_pts], mo[sel_pts]).mean()
            m = rand_masks[RNG.integers(0, n_rand_sim)]
            acc_rnd = np.where(m[sel_pts], fo[sel_pts], mo[sel_pts]).mean()
            diffs.append(acc_sel - acc_rnd)
        diffs = np.array(diffs)
        lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        lines.append(f"\n**选择性 vs 随机（paired bootstrap 1000 次）：Δ诚实率 = {diffs.mean()*100:.1f}pt "
                     f"[{lo_d*100:.1f}, {hi_d*100:.1f}]，p = {p:.4f}**")
    # 四格转移 + 误改率（全样本口径：输出中被改错的正确样本 / 全部正确样本）
    t = np.zeros((2, 2), int)
    for m, f in zip(mo, fo):
        t[1 - m, 1 - f] += 1
    lines.append("\n### 四格转移（全样本，真实修正后）")
    lines.append("| | 修正后错误 | 修正后正确 |")
    lines.append("|---|---|---|")
    lines.append(f"| 主回答错误 | {t[1,1]} | {t[1,0]} |")
    lines.append(f"| 主回答正确 | {t[0,1]} | {t[0,0]} |")
    wr_all = t[0,1] / max(t[0,0] + t[0,1], 1)
    lines.append(f"\n误改率（正确→错误）：**{wr_all*100:.1f}%**（{t[0,1]}/{t[0,0]+t[0,1]}）")
    hl = labels == 1
    lines.append("\n### 二项比例 95% CI（Wilson）")
    for nm3, v, nn in [("主回答诚实率", mo.mean(), n), ("修正后诚实率", fo.mean(), n),
                       ("幻觉样本修正后答对率", fo[hl].mean(), hl.sum())]:
        l, h = binom_ci(v, nn)
        lines.append(f"- {nm3}: {v*100:.1f}% [{l*100:.1f}%, {h*100:.1f}%]")
    # 阈值敏感性
    lines.append("\n### 阈值敏感性（离散 0/0.5/1）")
    lines.append("| 阈值 | 触发率 | 诚实率 | 错误修正率 | 误改率 | 调用/任务 |")
    lines.append("|---|---|---|---|---|---|")
    for th in [0.0, 0.5, 1.0]:
        mask = scores >= th
        acc, fix, wr = stratify(rs, mask, labels)
        lines.append(f"| {th} | {mask.mean()*100:.1f}% | {acc*100:.1f}% | {fix*100:.1f}% | {wr*100:.1f}% | {DET_CALLS+RED_CALLS*mask.mean():.2f} |")
    lines.append("\n- score 只取 0/0.5/1 三个离散值 → **离散风险等级**而非连续置信度")
    return "\n".join(lines), rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="full_reduce_cache.jsonl")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    d = json.load(open(RESULT / args.cache, encoding="utf-8"))
    rs = d["results"]
    labels = np.array([r["label"] for r in rs])
    pairs = [[2 * j, 2 * j + 1] for j in range(len(rs) // 2)]
    title = f"# 对照与统计可靠性分析 v2（2026-08-12，真实对照实验）\n\n数据：HaluEval 100 题 × 2 任务 = 200 任务，主 deepseek-v4-pro / 交叉 deepseek-v4-flash\n成本口径：检测 {DET_CALLS} 次调用/任务 + 触发修正 {RED_CALLS} 次（修正自 halu_loop.py 实际流程）"
    if args.offline:
        title = title.replace("真实对照实验", "离线模拟（未触发样本无真实修正结果，不能作为正式实验结论）")
    doc = [title]
    txt, rows = analyze("方案 A（跨模型重生成）", rs, labels, args.offline, pairs)
    doc.append(txt)
    doc.append("\n## 成本效益表")
    doc.append("| 方法 | 诚实率 | 调用/任务 | 相对裸答费用 |")
    doc.append("|---|---|---|---|")
    for nm2, acc_s, fix, wr, calls in rows:
        doc.append(f"| {nm2} | {acc_s} | {calls:.2f} | ×{calls/DET_CALLS:.2f} |")
    md = "\n".join(doc)
    (OUT / f"分析结果_{'real' if not args.offline else 'offline'}.md").write_text(md, encoding="utf-8")
    print(md)

if __name__ == "__main__":
    main()
