"""流式诊断：MMLU-Pro 剩余 3 条为什么空响应/卡死"""
import time
from llm_client import LLMClient
from bare_bench import load, sample, fmt_mmlupro

df, dev, letters = load("mmlupro")
df = sample(df, 500)
targets = {"3542", "6030", "10374"}
rows = [(str(r.get("question_id")), fmt_mmlupro(r["question"], r["options"]), str(r["answer"]))
        for _, r in df.iterrows() if str(r.get("question_id")) in targets]

c = LLMClient("deepseek/deepseek-v4-pro")
for qid, prompt, gold in rows:
    print(f"\n===== qid={qid} gold={gold} prompt_len={len(prompt)} =====", flush=True)
    t0 = time.time()
    try:
        stream = c.client.chat.completions.create(
            model=c.model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=40000, stream=True)
        got = {"reasoning": 0, "content": 0}
        finish, usage, first = None, None, None
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            r = getattr(d, "reasoning_content", None)
            ct = getattr(d, "content", None)
            if r:
                got["reasoning"] += len(r)
                if first is None:
                    first = ("reasoning", r[:50])
            if ct:
                got["content"] += len(ct)
                if first is None:
                    first = ("content", ct[:50])
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            if chunk.usage:
                usage = chunk.usage
        print(f"  耗时 {time.time()-t0:.1f}s | reasoning字符={got['reasoning']} content字符={got['content']} | finish={finish}")
        print(f"  首个delta: {first}")
        if usage:
            print(f"  usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
                  f"reasoning={getattr(getattr(usage,'completion_tokens_details',None),'reasoning_tokens',None)}")
    except Exception as e:
        print(f"  异常 {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:300]}", flush=True)
