"""抓取 pro 在卡死题上的完整思考内容（reasoning_content），存文件并打印开头"""
import time
from llm_client import LLMClient
from bare_bench import load, sample, fmt_mmlupro

df, dev, letters = load("mmlupro")
df = sample(df, 500)
rows = {str(r.get("question_id")): fmt_mmlupro(r["question"], r["options"])
        for _, r in df.iterrows() if str(r.get("question_id")) in {"3542"}}

c = LLMClient("deepseek/deepseek-v4-pro")
t0 = time.time()
buf_r, buf_c = [], []
try:
    stream = c.client.chat.completions.create(
        model=c.model, messages=[{"role": "user", "content": rows["3542"]}],
        temperature=0.0, max_tokens=40000, stream=True, timeout=600)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        r = getattr(d, "reasoning_content", None)
        ct = getattr(d, "content", None)
        if r:
            buf_r.append(r)
        if ct:
            buf_c.append(ct)
    reasoning = "".join(buf_r)
    content = "".join(buf_c)
    with open("results/pro_reasoning_3542.txt", "w", encoding="utf-8") as f:
        f.write(f"=== reasoning_content ({len(reasoning)} chars, {time.time()-t0:.1f}s) ===\n{reasoning}\n\n=== content ===\n{content}\n")
    print(f"完成 {time.time()-t0:.1f}s | reasoning={len(reasoning)} chars | content={repr(content[:60])}", flush=True)
    print("=== 思考开头 1200 字 ===", flush=True)
    print(reasoning[:1200], flush=True)
except Exception as e:
    print(f"异常 {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
