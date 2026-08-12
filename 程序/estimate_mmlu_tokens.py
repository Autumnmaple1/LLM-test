"""估算 MMLU / MMLU-Pro 实验的 token 消耗（重点：reasoning 部分）
方法：
1. 输入 prompt：按 bare_bench.py 同逻辑重建（分层抽样 + 5-shot 前缀），chars/4 估算
2. completion/reasoning：抽 calib-n 题实测每答均值（与 bare_bench 相同配置：
   temperature=0.0, max_tokens=40000, 默认思考模式），再外推到全量
3. 已跑实验（2026-08 中旬）都在 _log_usage 功能之前，无精确记录 → 本脚本输出为估算
用法: python3 estimate_mmlu_tokens.py [--calib-n 8] [--skip-calib]
"""
import argparse, json, time
from pathlib import Path
import pandas as pd

from bare_bench import load, sample, shot_prefix, fmt_mmlu, fmt_mmlupro, MODEL
from llm_client import get_client

CACHE = Path(__file__).parent / "results"
EXPS = [  # (名称, bench, 模型, 题数, shots)
    ("MMLU flash",      "mmlu",    "deepseek/deepseek-v4-flash", 456, 5),
    ("MMLU pro",        "mmlu",    "deepseek/deepseek-v4-pro",   500, 5),
    ("MMLU-Pro flash",  "mmlupro", "deepseek/deepseek-v4-flash", 490, 0),
    ("MMLU-Pro pro",    "mmlupro", "deepseek/deepseek-v4-pro",   490, 0),
]

def est_prompt_chars(bench, n, shots):
    """重建全量实验 prompt，统计字符数（与 bare_bench.run 相同构造）"""
    df, dev, letters = load(bench)
    df = sample(df, n)
    total_chars = 0
    for _, r in df.iterrows():
        if bench == "mmlu":
            pre = shot_prefix(dev, r["subject"], shots, bench)
            total_chars += len(pre + fmt_mmlu(r["question"], r["choices"], r["answer"]))
        else:
            total_chars += len(fmt_mmlupro(r["question"], r["options"]))
    return total_chars

def calib(bench, model, n_calib, shots):
    """抽 n_calib 题实测，返回每答 completion/reasoning 均值"""
    df, dev, letters = load(bench)
    df = sample(df, 500)
    if bench == "mmlu":
        per = max(1, 500 // df["subject"].nunique())
        idx = df.groupby("subject", group_keys=False).head(per).index
    else:
        per = max(1, 500 // df["category"].nunique())
        idx = df.groupby("category", group_keys=False).head(per).index
    sub = df.loc[idx].reset_index(drop=True).head(n_calib)
    c = get_client(model, temperature=0.0, max_tokens=40000)
    comps, reasons = [], []
    for _, r in sub.iterrows():
        if bench == "mmlu":
            pre = shot_prefix(dev, r["subject"], shots, bench)
            prompt = pre + fmt_mmlu(r["question"], r["choices"], r["answer"])
        else:
            prompt = fmt_mmlupro(r["question"], r["options"])
        resp = c.generate([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=40000) or ""
        # 从刚写的 usage 记录读本次调用
        recs = [json.loads(l) for l in (CACHE / "token_usage.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = recs[-1]
        comps.append(rec["completion_tokens"]); reasons.append(rec["reasoning_tokens"])
        print(f"  [{model.split('/')[-1]}] 校准题 {len(comps)}: comp={rec['completion_tokens']} reason={rec['reasoning_tokens']} (答: {resp[:40]!r})")
    return sum(comps)/len(comps), sum(reasons)/len(reasons)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-n", type=int, default=8)
    ap.add_argument("--skip-calib", action="store_true")
    args = ap.parse_args()

    print("== 1. 输入 prompt 估算（重建全量，chars/4）==")
    rows = []
    for name, bench, model, n, shots in EXPS:
        chars = est_prompt_chars(bench, n, shots)
        ptok = chars / 4
        rows.append({"name": name, "model": model, "n": n, "shots": shots, "prompt_chars": chars, "prompt_tok": ptok})
        print(f"  {name:<14} n={n:<4} shots={shots} 字符={chars:>10,} → 输入≈{ptok/1e4:6.1f}万 token")

    print("\n== 2. completion/reasoning 校准外推 ==")
    if args.skip_calib:
        print("  跳过实测（--skip-calib），只报输入估算")
    else:
        for r, (_, bench, _, _, _) in zip(rows, EXPS):
            try:
                comp_avg, reason_avg = calib(bench, r["model"], args.calib_n, r["shots"])
            except Exception as e:
                print(f"  {r['name']} 实测失败: {e}")
                comp_avg = reason_avg = None
            r["comp_avg"] = comp_avg
            r["reason_avg"] = reason_avg
            if comp_avg:
                r["comp_tot"] = comp_avg * r["n"]
                r["reason_tot"] = reason_avg * r["n"]
                print(f"  {r['name']:<14} 每答 comp={comp_avg:>7.0f} reason={reason_avg:>7.0f} → 全量 comp={r['comp_tot']/1e4:7.1f}万 reason={r['reason_tot']/1e4:7.1f}万")

    print("\n== 3. 汇总（估算值）==")
    print(f"{'实验':<14}{'输入':>10}{'completion':>12}{'reasoning':>12}{'reason占比':>10}")
    for r in rows:
        ct = r.get("comp_tot", 0); rt = r.get("reason_tot", 0)
        pct = f"{rt/ct*100:.1f}%" if ct else "-"
        print(f"{r['name']:<14}{r['prompt_tok']/1e4:>9.1f}万{ct/1e4:>11.1f}万{rt/1e4:>11.1f}万{pct:>10}")

if __name__ == "__main__":
    main()
