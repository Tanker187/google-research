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

"""Processes and combines results from initial and feedback rounds.

Supports multiple feedback rounds: the initial round and intermediate
feedback rounds use a strict filter (best_of_n == 1 and pass_check),
while the final feedback round uses a relaxed filter (best_of_n == 1).
"""

import argparse
import os

from agentic_question_gen_pipeline.utils import aggregate_bon_results
import pandas as pd


def _process_judge_results(file_path):
  """Loads judge results and computes pass_check per question.

  Args:
    file_path: Path to the JSONL file with judge results.

  Returns:
    DataFrame with aggregated results and pass_check column.
  """
  df = pd.read_json(file_path, lines=True)
  agg_df = aggregate_bon_results(
      df,
      search_trajectory_column="search_results",
      answer_column="answer",
      extra_column_to_keeps=["search_steps"],
  )
  agg_df["pass_check"] = agg_df.apply(
      lambda row: (
          row["best_of_n"] == 1
          and row["best_n_search_steps"] >= row["search_steps"]
      ),
      axis=1,
  )
  return agg_df


def main():
  parser = argparse.ArgumentParser(
      description="Process generated data from initial and feedback rounds."
  )
  parser.add_argument(
      "--initial_round_file",
      required=True,
      help="LLM judge results for the initial generation round",
  )
  parser.add_argument(
      "--feedback_round_files",
      nargs="+",
      default=[],
      help=(
          "LLM judge results for each feedback round "
          "(one or more files, in order)"
      ),
  )
  parser.add_argument(
      "--output_file_name",
      required=True,
      help="Output file base name (without extension)",
  )
  parser.add_argument(
      "--min_search_step",
      type=int,
      default=0,
      help="Filter out questions requiring fewer search steps",
  )
  args = parser.parse_args()

  output_dir = "outputs/agentic_question_gen/processed_data"
  os.makedirs(output_dir, exist_ok=True)

  all_passed = []

  # Process initial generation results (strict filter).
  print(">> Loading initial generation results from:", args.initial_round_file)
  agg_initial_df = _process_judge_results(args.initial_round_file)
  passed_initial_df = agg_initial_df.query(
      "best_of_n == 1 and pass_check"
  ).reset_index(drop=True)
  print(
      f">> Initial generation: {len(agg_initial_df)} total, "
      f"{len(passed_initial_df)} passed"
  )
  all_passed.append(
      passed_initial_df[["question", "answer", "best_n_search_steps"]]
  )

  # Process each feedback round.
  for i, feedback_file in enumerate(args.feedback_round_files):
    round_num = i + 1
    is_last_round = i == len(args.feedback_round_files) - 1
    print(
        f">> Loading feedback round {round_num} results from: {feedback_file}"
    )
    agg_feedback_df = _process_judge_results(feedback_file)

    if is_last_round:
      # Final round: keep all correctly answerable questions.
      passed_df = agg_feedback_df.query("best_of_n == 1").reset_index(drop=True)
      print(
          f">> Feedback round {round_num} (final): "
          f"{len(agg_feedback_df)} total, "
          f"{len(passed_df)} passed (best_of_n == 1)"
      )
    else:
      # Intermediate rounds: strict filter.
      passed_df = agg_feedback_df.query(
          "best_of_n == 1 and pass_check"
      ).reset_index(drop=True)
      print(
          f">> Feedback round {round_num}: "
          f"{len(agg_feedback_df)} total, "
          f"{len(passed_df)} passed "
          "(best_of_n == 1 and pass_check)"
      )

    all_passed.append(passed_df[["question", "answer", "best_n_search_steps"]])

  # Combine all rounds, deduplicate by question.
  combined_df = (
      pd.concat(all_passed, axis=0)
      .drop_duplicates(subset="question")
      .reset_index(drop=True)
  )
  print(f">> Combined: {len(combined_df)} unique questions")

  # Filter by minimum search steps.
  if args.min_search_step > 0:
    combined_df = combined_df.query(
        "best_n_search_steps >= @args.min_search_step"
    ).reset_index(drop=True)
    print(
        ">> After filtering "
        f"(min_search_step={args.min_search_step}): "
        f"{len(combined_df)} questions"
    )

  combined_df = combined_df[["question", "answer"]].reset_index(drop=True)

  # Save output.
  output_path = os.path.join(output_dir, f"{args.output_file_name}.jsonl")
  combined_df.to_json(output_path, orient="records", lines=True)
  print(f">> Saved output to: {output_path}")


if __name__ == "__main__":
  main()
