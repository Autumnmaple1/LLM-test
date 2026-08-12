import json
import re
import time
import threading

from openai import APITimeoutError, APIConnectionError, APIStatusError
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm

from llm_client import LLMClient, chat_memory

# ==================== 配置 ====================

# 要评测的科目列表，设为 None 表示评测全部 57 个科目
# 也可以指定几个科目快速测试，如: ["college_computer_science", "college_mathematics"]
SELECTED_SUBJECTS: list[str] | None = [
    "college_computer_science",
    "college_biology",
    "college_chemistry",
]  # None=全部

MAX_TEST_SAMPLES: int | None = None  # 每个科目最多评测多少题，None=全部（用于快速测试）
NUM_WORKERS = 10  # 并发线程数
RESULT_DIR = Path(__file__).parent / "results"

# 选项标签
CHOICE_LABELS = ["A", "B", "C", "D"]

_print_lock = threading.Lock()  # 线程安全打印


# ==================== Prompt 模板 ====================


def build_zero_shot_prompt(question: str, choices: list[str]) -> str:
    """构造零样本 prompt（无示例，直接问）"""
    parts = ["请回答以下单项选择题。只输出一个字母 (A/B/C/D)，不要输出任何其他内容。\n"]
    parts.append(question)
    for j, choice in enumerate(choices):
        parts.append(f"{CHOICE_LABELS[j]}. {choice}")
    parts.append("你的答案 (仅输出字母):")
    return "\n".join(parts)


def extract_answer(response: str) -> str | None:
    """从 LLM 回复中提取 A/B/C/D 答案"""
    text = response.strip()

    # 策略1: 匹配 "答案是 X" 或 "答案: X" 等中文模式
    m = re.search(r"答案[是为:：]\s*([A-Da-d])", text)
    if m:
        return m.group(1).upper()

    # 策略2: 匹配开头的单个 A/B/C/D
    m = re.match(r"^\s*([A-Da-d])\s*[\.\)、，,]", text)
    if m:
        return m.group(1).upper()

    # 策略3: 匹配单独成行的 A/B/C/D
    m = re.search(r"(?:^|\n)\s*([A-Da-d])\s*(?:$|[\n\.\)、，])", text)
    if m:
        return m.group(1).upper()

    # 策略4: 直接匹配开头的单个字母
    m = re.match(r"^\s*([A-Da-d])\s*$", text)
    if m:
        return m.group(1).upper()

    # 策略5: 在回复中搜索最后一个出现的选项字母
    m = re.findall(r"\b([A-D])\b", text)
    if m:
        return m[-1]

    return None


# ==================== 评测核心 ====================


def _eval_single_question(
    sample: dict,
    subject: str,
    total: int,
) -> dict:
    """单题评测（供线程池调用），每次创建独立 LLMClient，安全并发"""
    question = sample["question"]
    choices = sample["choices"]
    ground_truth = sample["answer"]
    correct_label = CHOICE_LABELS[ground_truth]
    idx = sample["_idx"]
    t_start = time.time()

    with _print_lock:
        print(f"  [开始] 第{idx}/{total}题 ...")

    prompt = build_zero_shot_prompt(question, choices)

    # 每个线程独立创建 LLMClient，无共享状态
    client = LLMClient()
    chat = chat_memory()
    chat.add_message("user", prompt)

    # 调 API + 提取答案，超时/异常/空响应/提取失败均重试（最多 10 次）
    response = ""
    predicted_label = None
    for attempt in range(10):
        try:
            response = client.generate_response(chat)
        except (APITimeoutError, APIConnectionError) as e:
            wait = min(2**attempt, 60)
            print(f"  [重试] 第{idx}题 {type(e).__name__}，{wait}s 后重试...")
            time.sleep(wait)
            continue
        except APIStatusError as e:
            wait = min(2**attempt, 60)
            print(f"  [重试] 第{idx}题 HTTP {e.status_code}，{wait}s 后重试...")
            time.sleep(wait)
            continue
        except Exception as e:
            wait = min(2**attempt, 60)
            print(f"  [重试] 第{idx}题异常 {type(e).__name__}: {e}，{wait}s 后重试...")
            time.sleep(wait)
            continue

        if not (response and response.strip()):
            print(f"  [重试] 第{idx}题空响应，重新请求...")
            continue
        predicted_label = extract_answer(response)
        if predicted_label is not None:
            break
        print(f"  [重试] 第{idx}题无法提取答案，重新请求...")
    else:
        # 10 次都没拿到有效答案，标记为失败
        response = response or ""
        predicted_label = None

    is_correct = predicted_label == correct_label
    elapsed = time.time() - t_start

    with _print_lock:
        status = "✓" if is_correct else "✗" if predicted_label is not None else "?"
        print(
            f"  [{status}] 第{idx}/{total}题  "
            f"预测={predicted_label or '?'}  正确={correct_label}  "
            f"耗时={elapsed:.1f}s"
        )

    return {
        "subject": subject,
        "question_idx": idx,
        "question": question,
        "choices": json.dumps(choices, ensure_ascii=False),
        "correct_answer": correct_label,
        "predicted_answer": predicted_label or "?",
        "is_correct": is_correct,
        "raw_response": response,
    }


def evaluate_subject(
    subject: str,
    max_samples: int | None = None,
    num_workers: int = NUM_WORKERS,
) -> list[dict]:
    """并发评测单个科目（zero-shot），返回按 question_idx 排序的结果列表

    Args:
        subject: 科目名称
        max_samples: 最多评测题数
        num_workers: 并发线程数
    """
    dataset = load_dataset("cais/mmlu", subject, split="test")
    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # 准备任务列表（附上索引以便排序）
    tasks = []
    for i in range(total):
        sample = dict(dataset[i])
        sample["_idx"] = i
        tasks.append(sample)

    results = []
    correct = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_map = {
            executor.submit(_eval_single_question, task, subject, total): task["_idx"]
            for task in tasks
        }

        pending = set(future_map.keys())
        t0 = time.time()

        with tqdm(total=total, desc=f"评测 {subject}", unit="题") as pbar:
            while pending:
                # 每 30s 汇报仍在运行的题目
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    if result["is_correct"]:
                        correct += 1
                    pbar.update(1)

                if pending:
                    pending_ids = sorted(future_map[f] for f in pending)
                    elapsed = time.time() - t0
                    with _print_lock:
                        print(
                            f"  [等待中] {elapsed:.0f}s | "
                            f"剩余 {len(pending)} 题: {pending_ids}"
                        )

    # 按原始顺序排序
    results.sort(key=lambda r: r["question_idx"])

    acc = correct / total * 100 if total > 0 else 0
    print(f"  {subject}: {correct}/{total} = {acc:.1f}%")

    return results


def print_summary(df: pd.DataFrame, title: str = ""):
    """打印并返回汇总统计"""
    if title:
        print(f"\n--- {title} ---")
    total_correct = df["is_correct"].sum()
    total_count = len(df)
    acc = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"总正确率: {total_correct}/{total_count} = {acc:.2f}%")

    subject_stats = df.groupby("subject").agg(
        总题数=("is_correct", "count"),
        正确数=("is_correct", "sum"),
        正确率=("is_correct", lambda x: f"{x.sum()/len(x)*100:.1f}%"),
    )
    print(subject_stats.to_string())
    return subject_stats


def run_benchmark(
    subjects: list[str] | None = None,
    max_samples: int | None = MAX_TEST_SAMPLES,
    result_dir: Path = RESULT_DIR,
):
    """主评测流程（zero-shot）"""
    result_dir.mkdir(parents=True, exist_ok=True)

    # 确定科目列表
    if subjects is None:
        all_configs = get_dataset_config_names("cais/mmlu")
        subjects = [s for s in all_configs if s not in ("all", "auxiliary_train")]
        print(f"共 {len(subjects)} 个科目待评测")
    else:
        print(f"指定 {len(subjects)} 个科目: {subjects}")

    # 打印模型信息
    temp_client = LLMClient()
    print(f"模型: {temp_client.model}")

    print(f"\n{'#'*60}")
    print(f"# 模式: zero_shot")
    print(f"{'#'*60}")

    all_results = []

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"科目: {subject}")

        results = evaluate_subject(subject, max_samples)
        all_results.extend(results)

        # 每个科目完成后保存中间结果
        df = pd.DataFrame(all_results)
        df.to_csv(
            result_dir / "mmlu_zero_shot.csv",
            index=False,
            encoding="utf-8-sig",
        )

    df = pd.DataFrame(all_results)

    # 汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    summary = print_summary(df)
    summary.to_csv(result_dir / "mmlu_zero_shot_summary.csv", encoding="utf-8-sig")

    # 保存错误题目
    wrong = df[~df["is_correct"]]
    if len(wrong) > 0:
        wrong.to_csv(
            result_dir / "mmlu_zero_shot_wrong.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"\n所有结果已保存至: {result_dir}")
    return df


# ==================== 入口 ====================

if __name__ == "__main__":
    run_benchmark(subjects=SELECTED_SUBJECTS)
