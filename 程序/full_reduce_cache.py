"""完整修正缓存：固定检测结果，对所有任务真实执行方案 A 修正（thresh=0 全触发）
输出 results/full_reduce_cache.jsonl：每个任务有真实修正 final + 重新 judge 的 final_ok
从它构造 全量/随机/选择性/Oracle 策略 = 真实对照实验（非反事实模拟）
用法: python3 full_reduce_cache.py
"""
import json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from halu_loop import load_halu, reduce_a, C, DS
from benchmark_truthfulqa_gen import llm_judge

RESULT = Path(__file__).parent / "results"

def main():
    old = json.load(open(RESULT / "halu_a_t0.5.jsonl", encoding="utf-8"))
    items = load_halu(100)
    judge = C(DS, 0.0, mtok=40000)
    tasks = [{"it": items[r["i"] // 2], "i": r["i"], "answer": r["main"], "label": r["label"],
              "score": r["score"], "main_ok": r["main_ok"]} for r in old["results"]]
    out, fails = [], 0
    t0 = time.time()
    def one(t):
        red = reduce_a(t["it"], t["answer"], t["score"], 0.0)  # thresh=0 → 全部真实执行修正
        ok = 1 if llm_judge(judge, t["it"]["question"], t["it"]["right_answer"], red["final"]) == "TRUE" else 0
        return t, red["final"], ok
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(one, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                t, final, ok = f.result()
                out.append({"i": t["i"], "label": t["label"], "score": t["score"],
                            "main_ok": t["main_ok"], "final_ok": ok,
                            "main": t["answer"], "final": final})
            except Exception as e:
                fails += 1
                print(f"[跳过] {type(e).__name__}")
    out.sort(key=lambda r: r["i"])
    json.dump({"main": old["main"], "ds": old["ds"], "n": len(out), "results": out},
              open(RESULT / "full_reduce_cache.jsonl", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"完成 {len(out)} 任务（跳过 {fails}），耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
