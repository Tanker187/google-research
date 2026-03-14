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

"""Generates improved question-answer pairs using execution feedback."""

import argparse
import concurrent.futures
import os

from google import genai
import pandas as pd
from prompt_templates.question_generation_with_feedback import (
    CORRECTNESS_FEEDBACK_PROMPT_NO_SEARCH,
)
from prompt_templates.question_generation_with_feedback import (
    DIFFICULTY_FEEDBACK_PROMPT_NO_SEARCH,
)
import tqdm

from . import utils


def construct_input_doc(initial_doc):
  r"""Formats a raw document into 'Title: ...\nContent: ...' format.

  Args:
    initial_doc: The raw document text.

  Returns:
    Formatted string with title and content.
  """
  title, text = initial_doc.split("\n", 1)
  return f"Title: {title}\nContent: {text.strip()}"


def question_gen_pipeline(client, model_name, prompt, max_turn):
  """Generates a question-answer pair from a feedback prompt.

  Args:
    client: The Gemini API client.
    model_name: Name of the model to use.
    prompt: The feedback prompt containing agent traces.
    max_turn: Maximum search turns (typically 0 for feedback).

  Returns:
    A tuple of (question, answer, generator_response).
  """
  if max_turn == 0:
    final_qa_pair, generator_response = utils.question_gen(
        client=client, model_name=model_name, prompt=prompt
    )
  else:
    final_qa_pair, generator_response = utils.agentic_question_gen(
        client=client,
        model_name=model_name,
        prompt=prompt,
        max_turn=max_turn,
    )

  if final_qa_pair:
    question, answer = final_qa_pair
    print(">> Generated question, answer:", question, answer)
    return question, answer, generator_response
  else:
    print(">> No question generated.")
    print(">> Generator response:", generator_response)
    return None, None, generator_response


def main():
  parser = argparse.ArgumentParser(
      description="Generate feedback-improved question-answer pairs"
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.5-flash",
  )
  parser.add_argument("--n_sample", type=int, default=0)
  parser.add_argument(
      "--input_file_path",
      type=str,
      required=True,
      help="LLM judge results for the round to generate feedback on",
  )
  parser.add_argument(
      "--previous_feedback_file",
      type=str,
      default=None,
      help="Previous feedback generation output (for multi-round)",
  )
  parser.add_argument(
      "--resample_prompt_file",
      type=str,
      default=None,
  )
  parser.add_argument("--retriever_name", type=str, default="e5")
  parser.add_argument(
      "--max_turn",
      type=int,
      default=20,
      help="Maximum search steps for the question generation agent",
  )
  parser.add_argument(
      "--max_num_workers",
      type=int,
      default=8,
      help="Maximum number of concurrent workers",
  )
  parser.add_argument("--output_name", type=str, default=None)
  args = parser.parse_args()

  project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
  location = os.environ.get("GOOGLE_CLOUD_LOCATION")
  if not project_id:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")
  if not location:
    raise ValueError("GOOGLE_CLOUD_LOCATION environment variable is not set")

  client = genai.Client(vertexai=True, project=project_id, location=location)

  # Process input file.
  input_df = pd.read_json(args.input_file_path, lines=True)
  agg_df = utils.aggregate_bon_results(
      input_df,
      search_trajectory_column="search_results",
      answer_column="answer",
      extra_column_to_keeps=["search_steps", "model_response"],
  )
  agg_df["pass_check"] = agg_df.apply(
      lambda row: (
          row["best_of_n"] == 1
          and row["best_n_search_steps"] >= row["search_steps"]
      ),
      axis=1,
  )

  # Extract search agent prompt/response from best rollout.
  agg_df["search_agent_prompt"] = agg_df["best_response"].map(
      lambda data: "<think>".join(data.split("<think>")[:3]).strip()
  )
  agg_df["search_agent_response"] = agg_df["best_response"].map(
      lambda data: "<think> "
      + "<think>".join(data.split("<think>")[3:]).strip()
  )

  # Extract data generator prompt/response.
  if args.previous_feedback_file:
    feedback_df = pd.read_json(args.previous_feedback_file, lines=True)
    agg_df["data_generator_agent_prompt"] = agg_df.join(
        feedback_df[["question", "prompt"]].set_index("question"),
        on="question",
        how="left",
        lsuffix="_",
    )["prompt"]
    agg_df["data_generator_agent_response"] = agg_df.join(
        feedback_df[["question", "model_response"]].set_index("question"),
        on="question",
        how="left",
        lsuffix="_",
    )["model_response"]
  else:
    agg_df["data_generator_agent_prompt"] = agg_df["model_response"].map(
        lambda data: ("<think>".join(data.split("<think>")[:3]).strip())
    )
    agg_df["data_generator_agent_response"] = agg_df["model_response"].map(
        lambda data: (
            "<think> " + "<think>".join(data.split("<think>")[3:]).strip()
        )
    )

  if args.resample_prompt_file:
    resample_df = pd.read_json(args.resample_prompt_file, lines=True)
    resample_df = (
        resample_df[["question", "prompt"]]
        .drop_duplicates("question")
        .reset_index(drop=True)
    )
    agg_df["prompt"] = agg_df.join(
        resample_df[["question", "prompt"]].set_index("question"),
        on="question",
        how="left",
        lsuffix="_",
    )["prompt"]

  if args.n_sample > 0:
    agg_df = agg_df.sample(n=args.n_sample, random_state=42).reset_index(
        drop=True
    )

  # Split into difficulty and correctness feedback cases.
  difficulty_df = agg_df.query("best_of_n == 1 and not pass_check").reset_index(
      drop=True
  )
  correctness_df = agg_df.query("best_of_n == 0").reset_index(drop=True)

  # Override max_turn to 0 for feedback (no search).
  args.max_turn = 0

  # Build feedback prompts.
  difficulty_df["prompt"] = difficulty_df.apply(
      lambda row: DIFFICULTY_FEEDBACK_PROMPT_NO_SEARCH.format(
          target_step=row["search_steps"],
          data_generator_agent_prompt=(row["data_generator_agent_prompt"]),
          data_generator_agent_response=(row["data_generator_agent_response"]),
          search_agent_prompt=row["search_agent_prompt"],
          search_agent_response=row["search_agent_response"],
      ),
      axis=1,
  )
  difficulty_df["feedback_type"] = "difficulty"

  correctness_df["prompt"] = correctness_df.apply(
      lambda row: CORRECTNESS_FEEDBACK_PROMPT_NO_SEARCH.format(
          target_step=row["search_steps"],
          data_generator_agent_prompt=(row["data_generator_agent_prompt"]),
          data_generator_agent_response=(row["data_generator_agent_response"]),
          search_agent_prompt=row["search_agent_prompt"],
          search_agent_response=row["search_agent_response"],
      ),
      axis=1,
  )
  correctness_df["feedback_type"] = "correctness"

  data_with_feedback = pd.concat(
      [difficulty_df, correctness_df], axis=0
  ).reset_index(drop=True)
  data_with_feedback["prev_question"] = data_with_feedback["question"]
  data_with_feedback["prev_answer"] = data_with_feedback["answer"]

  print(f">> Processing feedback for {len(data_with_feedback)} samples.")

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.max_num_workers
  ) as executor:
    future_to_idx = {
        executor.submit(
            question_gen_pipeline,
            client,
            args.model_name,
            row["prompt"],
            args.max_turn,
        ): idx
        for idx, row in data_with_feedback.iterrows()
    }
    all_responses = []
    for future in tqdm.tqdm(
        concurrent.futures.as_completed(future_to_idx),
        total=len(data_with_feedback),
    ):
      idx = future_to_idx[future]
      try:
        response = future.result()
        all_responses.append((idx, response))
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error processing index {idx}: {e}")
        all_responses.append((idx, (None, None, None)))

    all_responses = [
        response for _, response in sorted(all_responses, key=lambda x: x[0])
    ]

  data_with_feedback["question"] = [r[0] for r in all_responses]
  data_with_feedback["answer"] = [r[1] for r in all_responses]
  data_with_feedback["model_response"] = [r[2] for r in all_responses]

  if args.output_name:
    output_file_path = args.output_name
  else:
    input_file_name = os.path.basename(args.input_file_path)
    output_file_path = input_file_name.replace(".jsonl", "")

  output_file_path = (
      f"{output_file_path}_{args.model_name}_{args.max_turn}turn.jsonl"
  )
  output_path = f"outputs/agentic_question_gen/feedbacks/{output_file_path}"
  data_with_feedback.to_json(output_path, orient="records", lines=True)
  print(">> saved output to:", output_path)


if __name__ == "__main__":
  main()
