"""补充诊断：flash vs pro（流式/非流式）在 3 条卡死题上的行为"""
import time
from llm_client import LLMClient
from bare_bench import load, sample, fmt_mmlupro

df, dev, letters = load("mmlupro")
df = sample(df, 500)
targets = {"3542", "6030", "10374"}
rows = {str(r.get("question_id")): fmt_mmlupro(r["question"], r["options"])
        for _, r in df.iterrows() if str(r.get("question_id")) in targets}


def try_once(label, model, qid, stream, mtok, wait=240):
    c = LLMClient(model)
    t0 = time.time()
    try:
        r = c.client.chat.completions.create(
            model=c.model, messages=[{"role": "user", "content": rows[qid]}],
            temperature=0.0, max_tokens=mtok, stream=stream, timeout=wait)
        if stream:
            n, nxt = 0, None
            for ch in r:
                if ch.choices:
                    d = ch.choices[0].delta
                    t = getattr(d, "reasoning_content", None) or getattr(d, "content", None)
                    if t:
                        n += 1
                        if nxt is None:
                            nxt = t[:40]
            print(f"[{label}] qid={qid} 流式 {time.time()-t0:.1f}s | chunks={n} | 首个: {nxt}", flush=True)
        else:
            content = r.choices[0].message.content
            print(f"[{label}] qid={qid} 非流式 {time.time()-t0:.1f}s | content={repr(content[:50])}", flush=True)
    except Exception as e:
        print(f"[{label}] qid={qid} 异常 {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:120]}", flush=True)


for qid in targets:
    try_once("flash-流式", "deepseek/deepseek-v4-flash", qid, True, 40000, wait=120)
try_once("pro-非流式", "deepseek/deepseek-v4-pro", "3542", False, 40000, wait=240)
try_once("pro-非流式", "deepseek/deepseek-v4-pro", "6030", False, 40000, wait=240)
