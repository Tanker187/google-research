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

"""Evaluates search agent answers using LLM-as-a-judge."""

import argparse
import concurrent.futures
import os

from google import genai
from google.api_core import exceptions as api_exceptions
import pandas as pd
from prompt_templates.question_generation import LLM_AS_A_JUDGE_PROMPT_LIST
import tqdm
import utils

tqdm = tqdm.tqdm


def get_llm_as_a_judge_result(
    api_client,
    model_name,
    question,
    gold_answer,
    model_answer,
    dataset,
    temperature,
    max_output_tokens,
):
  """Calls the LLM judge to evaluate an answer.

  Args:
    api_client: The Gemini API client.
    model_name: Name of the judge model.
    question: The question being evaluated.
    gold_answer: The expected gold answer.
    model_answer: The model's answer to evaluate.
    dataset: Dataset name (affects judgement extraction).
    temperature: Sampling temperature.
    max_output_tokens: Maximum output tokens.

  Returns:
    A tuple of (judge_prompt, judge_response, judgement).
  """
  llm_as_a_judge_prompt = LLM_AS_A_JUDGE_PROMPT_LIST.format(
      question=question,
      model_answer=model_answer,
      gold_answer=gold_answer,
  )
  llm_as_a_judge_response = utils.call_gemini_api(
      client=api_client,
      model_name=model_name,
      prompt=llm_as_a_judge_prompt,
      temperature=temperature,
      max_output_tokens=max_output_tokens,
  )

  if dataset == "fanoutqa":
    judgement = llm_as_a_judge_response.strip()[-1]
  else:
    judgement = utils.extract_judgement(llm_as_a_judge_response)
  return llm_as_a_judge_prompt, llm_as_a_judge_response, judgement


def main():
  parser = argparse.ArgumentParser(
      description="Evaluate answers using LLM-as-a-judge"
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
      "--output_file_path",
      type=str,
      default=None,
  )
  parser.add_argument("--n_sample", type=int, default=0)
  parser.add_argument("--temperature", type=float, default=0)
  parser.add_argument("--max_output_tokens", type=int, default=512)
  parser.add_argument("--max_num_workers", type=int, default=8)
  parser.add_argument(
      "--gold_answer_column",
      default="golden_answers",
      type=str,
  )
  parser.add_argument(
      "--answer_column",
      default="model_answer",
      type=str,
  )
  parser.add_argument(
      "--question_column",
      default="question",
      type=str,
  )
  parser.add_argument("--dataset", type=str, default=None)
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
  else:
    raise FileNotFoundError(f"Unsupported file format: {args.input_file_path}")

  if args.n_sample > 0:
    input_df = input_df[: args.n_sample]

  input_df["gold_answer"] = input_df[args.gold_answer_column].map(
      lambda x: (
          ",".join([str(item) for item in x]) if isinstance(x, list) else str(x)
      )
  )

  all_responses = []
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.max_num_workers
  ) as executor:
    future_to_idx = {
        executor.submit(
            get_llm_as_a_judge_result,
            client,
            args.model_name,
            row[args.question_column],
            row["gold_answer"],
            row[args.answer_column],
            args.dataset,
            args.temperature,
            args.max_output_tokens,
        ): idx
        for idx, row in input_df.iterrows()
    }
    for future in tqdm(
        concurrent.futures.as_completed(future_to_idx),
        total=len(future_to_idx),
    ):
      idx = future_to_idx[future]
      try:
        response = future.result()
        all_responses.append((idx, response))
      except api_exceptions.GoogleAPIError as e:
        print(f"Error processing index {idx}: {e}")
        all_responses.append((idx, (None, None, None)))

  all_responses = [
      response for _, response in sorted(all_responses, key=lambda x: x[0])
  ]

  input_df["judgement"] = [r[2] for r in all_responses]
  input_df["llm_as_a_judge_response"] = [r[1] for r in all_responses]
  input_df["llm_as_a_judge_prompt"] = [r[0] for r in all_responses]

  if not args.output_file_path:
    input_file_name = os.path.basename(args.input_file_path)
    output_file_path = input_file_name.replace(".jsonl", "")
  else:
    output_file_path = args.output_file_path

  if args.n_sample > 0:
    output_file_path += f"_sample{args.n_sample}"
  output_file_path += ".jsonl"
  input_df.to_json(
      f"outputs/gemini_llm_judge/{output_file_path}",
      orient="records",
      lines=True,
  )


if __name__ == "__main__":
  main()
