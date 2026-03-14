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

"""Runs best-of-N agentic search inference on generated questions."""

import argparse
import concurrent.futures
import os
import re

from google import genai
from google.api_core import exceptions as google_exceptions
import pandas as pd
from prompt_templates.agentic_answer_prompt import GEMINI_PROMPT
import requests
import tqdm
import utils


tqdm = tqdm.tqdm
_SEARCH_TEMPLATE = (
    "\n{output_text}\n<information>{search_results}</information>\n"
)


def search(query, k=3):
  """Calls the local retrieval API."""
  payload = {"queries": [query], "topk": k, "return_scores": True}
  results = requests.post(
      "http://127.0.0.1:8000/retrieve", json=payload
  ).json()["result"]
  return results[0]


def _passages2string(retrieval_result):
  """Formats retrieval results into a numbered document string."""
  parts = []
  for idx, doc_item in enumerate(retrieval_result):
    content = doc_item["document"]["contents"]
    title = content.split("\n")[0]
    text = "\n".join(content.split("\n")[1:])
    parts.append(f"Doc {idx + 1}(Title: {title}) {text}")
  return "\n".join(parts)


def agentic_search(
    client,
    model_name,
    question,
    gold_answer,
    max_turn,
    top_k,
    thinking_budget=-1,
    temperature=0.0,
    top_p=0.95,
    max_output_tokens=65536,
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
      gold_answer, full_prompt).
  """
  if stop_sequences is None:
    stop_sequences = ["</search>"]

  final_answer = None
  prompt = GEMINI_PROMPT + " " + question
  num_turns = 0
  search_trajectory = []

  while not final_answer and num_turns < max_turn:
    num_turns += 1
    response_text = utils.call_gemini_api(
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
      final_answer = utils.extract_solution(response_text)

    if final_answer is None:
      if not response_text:
        print(f"Step {num_turns}: No response from the model")
        continue
      response_text += "</search>"
      search_query = re.search(
          r"<search>(.*?)</search>", response_text, re.DOTALL
      )
      if search_query:
        search_query = search_query.group(1).strip()
        print(f"Searching for: {search_query}")
        search_results_raw = search(search_query, top_k)
        search_results = _passages2string(search_results_raw)
        search_trajectory.append({
            "query": search_query,
            "results": search_results_raw,
        })
      else:
        search_results = ""
      prompt += _SEARCH_TEMPLATE.format(
          output_text=response_text,
          search_results=search_results,
      )

  prompt += str(response_text)
  print(f"Final answer: {final_answer}; gold answer: {gold_answer}")
  return (
      str(final_answer),
      search_trajectory,
      question,
      gold_answer,
      prompt,
  )


def main():
  parser = argparse.ArgumentParser(
      description="Run best-of-N agentic search inference"
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.0-flash",
  )
  parser.add_argument(
      "--input_file_path",
      type=str,
      required=True,
  )
  parser.add_argument(
      "--output_path_name",
      type=str,
      default=None,
  )
  parser.add_argument(
      "--retriever_name",
      type=str,
      default="e5",
  )
  parser.add_argument(
      "--question_column",
      type=str,
      default="question",
  )
  parser.add_argument(
      "--gold_answer_column",
      type=str,
      default="gold_answer",
  )
  parser.add_argument("--n_sample", type=int, default=0)
  parser.add_argument(
      "--max_turn",
      type=int,
      default=10,
      help="Maximum number of search turns",
  )
  parser.add_argument(
      "--top_k",
      type=int,
      default=3,
      help="Number of passages to retrieve per search query",
  )
  parser.add_argument(
      "--serp_api_key",
      type=str,
      default=None,
      help="API key for the SERP API",
  )
  parser.add_argument(
      "--correct_data_only",
      action="store_true",
      help="Only use questions with correct answers",
  )
  parser.add_argument(
      "--thinking_budget",
      type=int,
      default=0,
      help="Max tokens for thinking (-1 for dynamic)",
  )
  parser.add_argument(
      "--max_num_workers",
      type=int,
      default=8,
      help="Maximum number of concurrent workers",
  )
  parser.add_argument("--n_rollout", type=int, default=1)
  parser.add_argument(
      "--temperature",
      type=float,
      default=0.0,
      help="Sampling temperature",
  )
  args = parser.parse_args()

  project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
  location = os.environ.get("GOOGLE_CLOUD_LOCATION")
  if not project_id:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")
  if not location:
    raise ValueError("GOOGLE_CLOUD_LOCATION environment variable is not set")
  client = genai.Client(vertexai=True, project=project_id, location=location)

  # Read input file.
  if args.input_file_path.endswith(".jsonl"):
    input_df = pd.read_json(args.input_file_path, lines=True)
  elif args.input_file_path.endswith(".json"):
    input_df = pd.read_json(args.input_file_path)
  elif args.input_file_path.endswith(".parquet"):
    input_df = pd.read_parquet(args.input_file_path)
  else:
    raise FileNotFoundError(f"Unsupported file format: {args.input_file_path}")

  if args.question_column not in input_df.columns:
    raise ValueError(f"Question column '{args.question_column}' not found.")
  if args.gold_answer_column not in input_df.columns:
    raise ValueError(
        f"Gold answer column '{args.gold_answer_column}' not found."
    )

  if args.correct_data_only:
    if "extracted_judgement" not in input_df.columns:
      raise ValueError("Input must contain 'extracted_judgement' column.")
    input_df = input_df.query("extracted_judgement == 'yes'").reset_index(
        drop=True
    )

  if args.n_sample > 0:
    input_df = input_df[: args.n_sample]

  if args.n_rollout > 1:
    print(f"Running {args.n_rollout} rollouts per question.")
    input_df = input_df.loc[input_df.index.repeat(args.n_rollout)].reset_index(
        drop=True
    )

  if args.retriever_name == "serp" and args.serp_api_key is None:
    raise ValueError("Please provide the SERP API key via --serp_api_key")

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.max_num_workers
  ) as executor:
    future_to_idx = {
        executor.submit(
            agentic_search,
            client,
            args.model_name,
            row[args.question_column],
            row[args.gold_answer_column],
            args.max_turn,
            args.top_k,
            args.thinking_budget,
            args.temperature,
        ): idx
        for idx, row in input_df.iterrows()
    }
    all_responses = []
    for future in tqdm(
        concurrent.futures.as_completed(future_to_idx),
        total=len(input_df),
    ):
      idx = future_to_idx[future]
      try:
        response = future.result(timeout=60)
        all_responses.append((idx, response))
      except (
          concurrent.futures.TimeoutError,
          requests.exceptions.RequestException,
          google_exceptions.GoogleAPIError,
      ) as e:
        print(f"Error processing row {idx}: {e}")
        all_responses.append((idx, (None, None, None, None, None)))

    all_responses = [
        response for _, response in sorted(all_responses, key=lambda x: x[0])
    ]

  input_df["prompt"] = [r[4] for r in all_responses]
  input_df["model_answer"] = [r[0] for r in all_responses]
  input_df["search_results"] = [r[1] for r in all_responses]
  input_df["gold_answer"] = [r[3] for r in all_responses]

  if not args.output_path_name:
    input_file_name = os.path.basename(args.input_file_path)
    output_file_path = input_file_name.replace(".jsonl", "")
  else:
    output_file_path = args.output_path_name

  if args.n_sample > 0:
    output_file_path += f"_sample{args.n_sample}"
  output_file_path += (
      f"_{args.model_name}_{args.retriever_name}"
      f"_top{args.top_k}_{args.max_turn}turn_new"
      f"_{args.n_rollout}_rollouts"
      f"_{args.thinking_budget}_thinking.jsonl"
  )
  print(f"Saving results to {output_file_path}")
  input_df.to_json(
      f"outputs/agentic_search/{output_file_path}",
      orient="records",
      lines=True,
  )


if __name__ == "__main__":
  main()
