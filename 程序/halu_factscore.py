"""HaluEval FActScore 对比（并行）：触发修正的幻觉样本，主回答 vs 最终回答的事实准确率（knowledge 为参考）"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from factscore import decompose_claims, verify_claims
from llm_client import get_client

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

def main(cache_file, max_n=40, workers=12):
    d = json.loads((RESULT_DIR / cache_file).read_text(encoding="utf-8"))
    items = [json.loads(l) for l in open(Path(__file__).parent / "data_halu" / "qa_data.json", encoding="utf-8") if l.strip()]
    rs = [r for r in d["results"] if r["triggered"] and r["label"] == 1][:max_n]
    if not rs:
        print("没有触发的幻觉样本")
        return
    c = get_client(PRO, temperature=0.0)
    jc = get_client(FLASH, temperature=0.0)

    def one(r):
        it = items[r["i"] // 2]
        fs_o, n_o = factscore(c, jc, r["main"], it["knowledge"])
        fs_n, n_n = factscore(c, jc, r["final"], it["knowledge"])
        return r["i"], fs_o, fs_n, n_o, n_n

    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, r) for r in rs]
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:
                print(f"  [跳过] 样本失败: {type(e).__name__}")
    if not out:
        print("全部失败")
        return
    out.sort()
    print(f"对 {len(out)} 条触发修正的幻觉样本计算 FActScore (knowledge 为参考)...")
    for i, fs_o, fs_n, n_o, n_n in out:
        print(f"  i={i} FActScore {fs_o:.2f} -> {fs_n:.2f} (claims {n_o}->{n_n})")
    avg_o = sum(x[1] for x in out) / len(out)
    avg_n = sum(x[2] for x in out) / len(out)
    up = sum(1 for x in out if x[2] > x[1])
    print(f"\nFActScore 均值: {avg_o*100:.1f}% -> {avg_n*100:.1f}% | 提升 {up}/{len(out)} 条")
    d["factscore"] = {"n": len(out), "main": avg_o, "final": avg_n, "improved": up / len(out)}
    (RESULT_DIR / cache_file).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "halu_t0.5.jsonl")
