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

"""Converts processed QA data to parquet format for RL training."""

import argparse
import os

import datasets
import pandas as pd


def make_prefix(dp, template_type):
  """Creates the prompt prefix for a data point.

  Args:
    dp: A dict-like data point with a 'question' field.
    template_type: The template type to use ('base' only).

  Returns:
    A string representing the prompt prefix.
  """
  question = dp["question"]
  if template_type == "base":
    prefix = (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> "
        "first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, "
        "you can call a search engine by <search> query </search>"
        " and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you "
        "can directly provide the answer inside <answer> and "
        "</answer>, without detailed illustrations. For example,"
        " <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )
  else:
    raise NotImplementedError
  return prefix


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", required=True)
  parser.add_argument(
      "--in_domain_test_dataset",
      type=str,
      default=(
          "./data/10turn_2to10_steps_gemini-2.5-flash_e5_top3"
          "_10turn_best_of_2_to_8_100_samples_each.parquet"
      ),
  )
  parser.add_argument(
      "--question_column",
      type=str,
      default="question",
  )
  parser.add_argument(
      "--answer_column",
      type=str,
      default="answer",
  )
  parser.add_argument(
      "--local_dir",
      default="./data/gemini_agentic_question_gen",
  )
  parser.add_argument(
      "--template_type",
      type=str,
      default="base",
  )
  parser.add_argument(
      "--dataset_name",
      type=str,
      required=True,
  )
  parser.add_argument(
      "--eval_dataset",
      type=str,
      default=None,
  )
  args = parser.parse_args()

  dataset_df = pd.read_json(args.input_file, lines=True)
  train_df = dataset_df

  if "data_source" not in train_df.columns:
    train_df["data_source"] = "synthetic"
  train_dataset = datasets.Dataset.from_pandas(
      train_df[["question", args.answer_column, "data_source"]]
  )

  def make_map_fn(split):
    def process_fn(example, idx):
      example["question"] = example[args.question_column].strip()
      if example["question"][-1] != "?":
        example["question"] += "?"
      question = make_prefix(example, template_type=args.template_type)
      answer_val = example[args.answer_column]
      solution = {
          "target": [answer_val] if isinstance(answer_val, str) else answer_val,
      }
      return {
          "data_source": example["data_source"],
          "prompt": [{"role": "user", "content": question}],
          "ability": "fact-reasoning",
          "reward_model": {
              "style": "rule",
              "ground_truth": solution,
          },
          "extra_info": {"split": split, "index": idx},
      }

    return process_fn

  train_dataset = train_dataset.map(
      function=make_map_fn("train"), with_indices=True
  )
  test_dataset = datasets.Dataset.from_pandas(
      pd.read_parquet(args.in_domain_test_dataset)
  )

  if args.eval_dataset is not None:
    eval_df = pd.read_json(args.eval_dataset, lines=True)
    eval_df[args.answer_column] = eval_df["gold_answer"].map(
        lambda data: (
            [str(item) for item in data]
            if isinstance(data, list)
            else [str(data)]
        )
    )
    eval_dataset = datasets.Dataset.from_pandas(
        eval_df[["question", "data_source", args.answer_column]]
    ).map(function=make_map_fn("test"), with_indices=True)
    test_dataset = datasets.concatenate_datasets([test_dataset, eval_dataset])

  local_dir = args.local_dir
  train_dataset.to_parquet(
      os.path.join(local_dir, f"{args.dataset_name}_train.parquet")
  )
  if args.eval_dataset is not None:
    test_dataset.to_parquet(
        os.path.join(
            local_dir,
            f"{args.dataset_name}_test_combined.parquet",
        )
    )
  else:
    test_dataset.to_parquet(
        os.path.join(local_dir, f"{args.dataset_name}_test.parquet")
    )

  print(">> Output train dataset size:", len(train_dataset))
  print(">> Output test dataset size:", len(test_dataset))


if __name__ == "__main__":
  main()
