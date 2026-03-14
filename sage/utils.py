# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utility functions for Gemini API calls, answer evaluation, and analysis."""

import collections
import random
import re
import string

from google.genai import types
import numpy as np
import pandas as pd
import requests
import tenacity
from tenacity import retry

wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt


@retry(
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(10),
)
def call_gemini_api(
    client,
    model_name,
    prompt,
    temperature=0.0,
    top_p=0.95,
    max_output_tokens=512,
    thinking_budget=-1,
    stop_sequences=None,
    print_response=False,
):
  """Calls the Gemini API to generate content.

  Args:
    client: The Gemini API client.
    model_name: Name of the model to use.
    prompt: The input prompt string.
    temperature: Sampling temperature.
    top_p: Top-p sampling parameter.
    max_output_tokens: Maximum number of output tokens.
    thinking_budget: Token budget for thinking. -1 for dynamic.
    stop_sequences: List of stop sequences.
    print_response: Whether to print the response.

  Returns:
    The generated text response.
  """
  if stop_sequences is None:
    stop_sequences = []

  if model_name == "gemini-2.0-flash":
    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            top_p=top_p,
        ),
    )
  else:
    if thinking_budget == 0 and model_name == "gemini-2.5-pro":
      thinking_budget = 128
    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            top_p=top_p,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget
            ),
        ),
    )

  response = chat.send_message(prompt).text
  if print_response:
    print(f"Response: {response}")
  return response


def extract_solution(solution_str):
  """Extracts the last <answer>...</answer> content from a string.

  Args:
    solution_str: The string containing the solution.

  Returns:
    The extracted solution string, or None if no <answer> tag is found.
  """
  matches = list(
      re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
  )
  if not matches:
    return None
  return matches[-1].group(1).strip()


def extract_judgement(response):
  """Extracts the 'correct: yes/no' judgement from an LLM judge response."""
  match = re.search(r"correct:\s*(yes|no)", response, re.IGNORECASE)
  if match:
    return match.group(1).strip().lower()
  return None


def local_search_api(query, k=5, url="http://127.0.0.1:8000/retrieve"):
  """Calls the local search API to retrieve documents.

  Args:
    query: The search query string.
    k: Number of top results to retrieve.
    url: The retrieval API endpoint URL.

  Returns:
    A list of retrieved document results.
  """
  payload = {"queries": [query], "topk": k, "return_scores": True}
  results = requests.post(url, json=payload).json()["result"]
  return results[0]


def format_retrieved_documents(documents):
  """Formats retrieved documents into a readable string."""
  parts = []
  for doc in documents:
    title = doc["document"]["contents"].split("\n")[0]
    text = "\n".join(doc["document"]["contents"].split("\n")[1:])
    parts.append(f"Title: {title}\nContent: {text}\n")
  return "\n".join(parts)


def normalize_answer(s):
  """Normalizes an answer string for evaluation comparison."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction, golden_answers):
  """Checks exact match between prediction and any golden answer."""
  if isinstance(golden_answers, str):
    golden_answers = [golden_answers]
  normalized_prediction = normalize_answer(prediction)
  for golden_answer in golden_answers:
    if normalize_answer(golden_answer) == normalized_prediction:
      return 1
  return 0


def subem_check(prediction, golden_answers):
  """Checks if any golden answer is a substring of the prediction."""
  if isinstance(golden_answers, str):
    golden_answers = [golden_answers]
  normalized_prediction = normalize_answer(prediction)
  for golden_answer in golden_answers:
    if normalize_answer(golden_answer) in normalized_prediction:
      return 1
  return 0


def get_tokens(s):
  """Tokenizes a normalized answer string."""
  if not s:
    return []
  return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
  """Computes token-level F1 score between gold and predicted answers."""
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  return (2 * precision * recall) / (precision + recall)


def compute_f1s(prediction, gold_answers):
  """Computes the max F1 score across all gold answers."""
  if isinstance(gold_answers, str):
    gold_answers = [gold_answers]
  return max(
      compute_f1(gold_answer, prediction) for gold_answer in gold_answers
  )


def process_question(question_str):
  """Ensures a question string ends with a question mark."""
  question = str(question_str).strip()
  if question[-1] != "?":
    question += "?"
  return question


def parse_reasoning_steps(text):
  """Parses bracketed items from each line of a reasoning steps string.

  Args:
    text: A string where each line contains items in the format '- Step X:
      [item1, item2]'.

  Returns:
    A list of lists, where each inner list contains the items
    from one line.
  """
  final_list = []
  for line in text.strip().split("\n"):
    match = re.search(r"\[(.*?)\]", line)
    if match:
      items = [item.strip() for item in match.group(1).split(",")]
      final_list.append(items)
  return final_list


def aggregate_bon_results(
    df,
    n=-1,
    search_trajectory_column="search_steps",
    answer_column="answer",
    extra_column_to_keeps=None,
):
  """Aggregates best-of-N rollout results per question.

  Groups results by question, selects the best correct rollout (fewest
  search steps among correct answers), and computes aggregate metrics.

  Args:
    df: DataFrame with rollout results including 'question', 'judgement',
      'model_answer', and 'prompt' columns.
    n: Max rollouts to consider per question. -1 for all.
    search_trajectory_column: Column with search trajectory data.
    answer_column: Column containing gold answers.
    extra_column_to_keeps: Additional columns to preserve from the first row of
      each question group.

  Returns:
    DataFrame with one row per question containing aggregate metrics.
  """
  if extra_column_to_keeps is None:
    extra_column_to_keeps = []

  questions = df["question"].dropna().unique()
  results = []
  for question in questions:
    question_df = df[df["question"] == question]
    if n > 0 and len(question_df) > n:
      question_df = question_df.sample(n=n, random_state=42)

    judgements = question_df["judgement"].tolist()
    n_search_steps = (
        question_df[search_trajectory_column]
        .map(lambda data: len(data) if data else 0)
        .tolist()
    )
    random_idx = np.random.randint(0, len(question_df))
    right_idx = [idx for idx, j in enumerate(judgements) if j == "yes"]
    best_idx = random.choice(right_idx) if right_idx else random_idx
    answer = (
        question_df[answer_column].iloc[0]
        if answer_column in question_df.columns
        else None
    )

    if right_idx:
      best_response_idx = np.argmin([n_search_steps[idx] for idx in right_idx])
      best_response_idx = right_idx[best_response_idx]
    else:
      non_null_idx = [
          idx
          for idx, ans in enumerate(question_df["model_answer"].tolist())
          if ans and ans.lower() != "none"
      ]
      if non_null_idx:
        best_response_idx = random.choice(non_null_idx)
      else:
        best_response_idx = random_idx

    result = {
        "question": question,
        "answer": answer,
        "model_answer": question_df["model_answer"].tolist(),
        "n_rollouts": len(question_df),
        "best_of_n": 1 if judgements[best_idx] == "yes" else 0,
        "random_of_n": 1 if judgements[random_idx] == "yes" else 0,
        "random_n_search_steps": n_search_steps[random_idx],
        "avg_of_n": np.mean([1 if j == "yes" else 0 for j in judgements]),
        "best_n_search_steps": n_search_steps[best_response_idx],
        "best_response": question_df["prompt"].iloc[best_response_idx],
        "best_idx": best_response_idx,
        "n_correct": len(right_idx),
    }
    for col in extra_column_to_keeps:
      if col in question_df.columns:
        result[col] = question_df[col].iloc[0]
    results.append(result)

  return pd.DataFrame(results)
