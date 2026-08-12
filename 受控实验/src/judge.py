"""判分：MMLU-Pro/TruthfulQA 字母匹配；HLE mc 字母、em 归一化匹配 + LLM judge 兜底"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from controlled_exp.src.client import ExpClient

def letter_of(text, letters="ABCDEFGHIJ"):
    m = re.search(r"\b([A-J])\b", str(text).upper())
    return m.group(1) if m else None

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def judge_equiv(client, question, ref, cand):
    """LLM 判分：候选答案与参考答案事实等价？返回 True/False"""
    p = ("Are the following two answers factually equivalent for the question? Answer YES or NO only.\n"
         f"Question: {question}\nReference answer: {ref}\nCandidate answer: {cand}\nEquivalent:")
    r = client.gen(p, thinking=False)
    return "YES" in r["resp"].upper()

def judge_mc(dataset, item, resp):
    """选择题判分，返回 (correct: bool, pred: str)"""
    if dataset == "truthfulqa":
        letters = "ABCDEFGHIJ"
        correct = item["labels"].index(1)
        pred = letter_of(resp, letters)
        return pred == letters[correct], pred
    if dataset == "mmlupro":
        pred = letter_of(resp)
        return pred == item["answer"], pred
    if dataset == "hle":
        pred = letter_of(resp)
        return pred == item["answer"], pred
    raise ValueError(dataset)

def judge_em(item, resp, client=None):
    """HLE exactMatch 判分：归一化匹配，不符走 LLM judge"""
    if norm(resp) == norm(item["answer"]):
        return True, "exact"
    if client is not None:
        return judge_equiv(client, item["question"], item["answer"], resp), "llm"
    return False, "none"

def judge(dataset, item, resp, client=None):
    if dataset == "hle" and item["answer_type"] == "exactMatch":
        return judge_em(item, resp, client)
    return judge_mc(dataset, item, resp)
