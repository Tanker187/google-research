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

"""Utility functions for agentic question generation pipeline."""

import random
import re

from google.genai import types
import numpy as np
import pandas as pd
from prompt_templates.agentic_answer_prompt import GEMINI_PROMPT
from prompt_templates.question_generation import LLM_AS_A_JUDGE_PROMPT
import requests
import tenacity

_SEARCH_TEMPLATE = (
    "\n{output_text}\n<information>{search_results}</information>\n"
)


@tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=2, max=60),
    stop=tenacity.stop_after_attempt(10),
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


def extract_qa_solution(solution_str):
  """Extracts the last <question>...<answer>...</answer> pair from a string.

  Args:
    solution_str: The input string containing the potential QA pair.

  Returns:
    A (question, answer) tuple, or None if no valid pair is found.
  """
  matches = list(
      re.finditer(r"<question>(.*?)</answer>", solution_str, re.DOTALL)
  )
  if not matches:
    return None

  question_answer_pair = matches[-1].group(1).strip()
  question = question_answer_pair.split("</question>")[0].strip()

  if "<answer>" not in question_answer_pair:
    return None
  parts = question_answer_pair.split("<answer>")
  if len(parts) < 2:
    return None

  answer = parts[1].strip()
  return question, answer


def extract_answer(solution_str):
  """Extracts the last <answer>...</answer> content from a string."""
  matches = list(
      re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
  )
  if not matches:
    return None
  return matches[-1].group(1).strip()


def extract_answering_steps(solution_str):
  """Extracts the last <answering step>...</answering step> content."""
  matches = list(
      re.finditer(
          r"<answering step>(.*?)</answering step>",
          solution_str,
          re.DOTALL,
      )
  )
  if not matches:
    return None
  return matches[-1].group(1).strip()


def search(query, k=3):
  """Calls the local retrieval API.

  Args:
    query: The search query string.
    k: Number of top results to retrieve.

  Returns:
    A list of retrieved document results.
  """
  payload = {"queries": [query], "topk": k, "return_scores": True}
  results = requests.post(
      "http://127.0.0.1:8000/retrieve", json=payload
  ).json()["result"]
  return results[0]


def passages2string(retrieval_result):
  """Formats retrieval results into a numbered document string."""
  parts = []
  for idx, doc_item in enumerate(retrieval_result):
    content = doc_item["document"]["contents"]
    title = content.split("\n")[0]
    text = "\n".join(content.split("\n")[1:])
    parts.append(f"Doc {idx + 1}(Title: {title}) {text}")
  return "\n".join(parts)


def extract_judgement(response):
  """Extracts 'correct: yes/no' from an LLM judge response."""
  match = re.search(r"correct:\s*(yes|no)", response, re.IGNORECASE)
  if match:
    return match.group(1).strip().lower()
  return None


def agentic_search(
    client,
    model_name,
    question,
    gold_answer,
    max_turn,
    top_k=3,
    thinking_budget=0,
    temperature=1.0,
    top_p=0.95,
    max_output_tokens=4096,
    stop_sequences=None,
):
  """Runs multi-turn agentic search to answer a question.

  Args:
    client: The Gemini API client.
    model_name: Name of the model to use.
    question: The question to answer.
    gold_answer: The expected gold answer.
    max_turn: Maximum number of search turns.
    top_k: Number of documents to retrieve per search.
    thinking_budget: Token budget for thinking.
    temperature: Sampling temperature.
    top_p: Top-p sampling parameter.
    max_output_tokens: Maximum output tokens per turn.
    stop_sequences: List of stop sequences.

  Returns:
    A tuple of (final_answer, search_trajectory, question,
    gold_answer, full_prompt, num_turns).
  """
  if stop_sequences is None:
    stop_sequences = ["</search>"]

  final_answer = None
  prompt = GEMINI_PROMPT + " " + question
  print(
      f">> Starting agentic search with question {question} "
      f"for {max_turn} turns."
  )
  num_turns = 0
  search_trajectory = []
  response_text = ""

  while not final_answer and num_turns < max_turn:
    response_text = call_gemini_api(
        client,
        model_name,
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        stop_sequences=stop_sequences,
        thinking_budget=thinking_budget,
    )
    if response_text:
      final_answer = extract_answer(response_text)

    if final_answer is None:
      if not response_text:
        print(f"Step {num_turns}: No response from the model")
        prompt += "<think>"
        continue
      response_text += "</search>"
      search_query = re.search(
          r"<search>(.*?)</search>", response_text, re.DOTALL
      )
      if search_query:
        search_query = search_query.group(1).strip()
        print(f"Searching for: {search_query}")
        search_results_raw = search(search_query, top_k)
        search_results = passages2string(search_results_raw)
        search_trajectory += search_results_raw
      else:
        search_results = ""
      prompt += _SEARCH_TEMPLATE.format(
          output_text=response_text,
          search_results=search_results,
      )
      num_turns += 1

  prompt += str(response_text)
  print(f"Final answer: {final_answer}; gold answer: {gold_answer}")
  return (
      str(final_answer),
      search_trajectory,
      question,
      gold_answer,
      prompt,
      num_turns,
  )


def question_gen(client, model_name, prompt):
  """Generates a question-answer pair without search.

  Args:
    client: The Gemini API client.
    model_name: Name of the model to use.
    prompt: The input prompt string.

  Returns:
    A tuple of ((question, answer), full_response) or
    (None, response) if extraction fails.
  """
  print(">> Starting question generation without search.")
  response = call_gemini_api(
      client=client,
      model_name=model_name,
      prompt=prompt,
      max_output_tokens=65535,
      temperature=1.0,
      thinking_budget=0,
  )
  if response:
    final_qa_pair = extract_qa_solution(response)
    return (final_qa_pair, response) if final_qa_pair else (None, response)
  return None, None


def agentic_question_gen(client, model_name, prompt, max_turn):
  """Generates a question-answer pair using multi-turn search.

  Iteratively searches and reasons to construct a complex question
  that requires multiple search steps to answer.

  Args:
    client: The Gemini API client.
    model_name: Name of the model to use.
    prompt: The initial generation prompt.
    max_turn: Maximum number of search turns.

  Returns:
    A tuple of ((question, answer), full_prompt) or
    (None, prompt) if generation fails.
  """
  final_qa_pair = None
  num_turns = 0
  output_trajectory = [prompt]
  print(f">> Starting question generation for {max_turn} turns.")
  response = ""

  while not final_qa_pair and num_turns < max_turn:
    num_turns += 1
    response = call_gemini_api(
        client=client,
        model_name=model_name,
        prompt=prompt,
        stop_sequences=["</search>"],
        max_output_tokens=65535,
        temperature=1.0,
        thinking_budget=0,
    )
    if not response:
      print(f"Step {num_turns}: No response, stopping.")
      break

    final_qa_pair = extract_qa_solution(response)
    if final_qa_pair is None:
      response += "</search>"
      search_query = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
      if search_query:
        search_query = search_query.group(1).strip()
        print("Searching for:", search_query)
        retrieved_docs = search(search_query)
        search_text = _SEARCH_TEMPLATE.format(
            output_text=response,
            search_results=passages2string(retrieved_docs),
        )
        output_trajectory.append(search_text)
        prompt += search_text
      else:
        break

  if response:
    prompt += response
    if not final_qa_pair:
      print("Forcing the model to generate a question.")
      force_prompt = (
          "\n<think>I have used up all the search budget and "
          "I will use the existing information to formulate a "
          "new plan and generate the question, answer, and "
          "answering plans."
      )
      new_prompt = "".join(output_trajectory) + force_prompt
      response = call_gemini_api(
          client=client,
          model_name=model_name,
          prompt=new_prompt,
          stop_sequences=["</search>"],
          max_output_tokens=65535,
          temperature=1.0,
          thinking_budget=0,
      )
      if response:
        final_qa_pair = extract_qa_solution(response)
        prompt = new_prompt + response

  return final_qa_pair, prompt


def verify_generated_qa(client, question, gold_answer, model_answer):
  """Verifies a generated QA pair using LLM-as-a-judge.

  Args:
    client: The Gemini API client.
    question: The generated question.
    gold_answer: The expected gold answer for the question.
    model_answer: The answer generated by the model.

  Returns:
    The judgement string ('yes' or 'no'), or empty string on failure.
  """
  judgement = ""
  judgement_prompt = LLM_AS_A_JUDGE_PROMPT.format(
      question=question,
      model_answer=model_answer,
      gold_answer=gold_answer,
  )
  llm_response = call_gemini_api(
      client=client,
      model_name="gemini-2.0-flash",
      prompt=judgement_prompt,
      temperature=0.0,
  )
  if llm_response:
    print(f"LLM as a judge response: {llm_response}")
    judgement = extract_judgement(llm_response)
  return judgement


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
    df: DataFrame with rollout results.
    n: Max rollouts to consider per question. -1 for all.
    search_trajectory_column: Column with search trajectory data.
    answer_column: Column containing gold answers.
    extra_column_to_keeps: Additional columns to preserve.

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
