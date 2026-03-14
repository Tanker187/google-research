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

"""Generates initial question-answer pairs from Wikipedia corpus documents."""

import argparse
import concurrent.futures
import os
import random

from agentic_question_gen_pipeline import utils
import datasets
from google import genai
import pandas as pd
from prompt_templates.question_generation_with_feedback import QUESTION_GENERATION_WITH_SEARCH
import tqdm


_PROMPT_MAP = {
    "with_search": QUESTION_GENERATION_WITH_SEARCH,
}


def construct_input_doc(initial_doc):
  r"""Formats a raw document into 'Title: ...\nContent: ...' format."""
  title, text = initial_doc.split("\n", 1)
  return f"Title: {title}\nContent: {text.strip()}"


def question_gen_pipeline(
    client,
    model_name,
    initial_doc,
    max_turn,
    search_step_min,
    search_step_max,
):
  """Generates a question-answer pair from an initial document.

  Args:
      client: The Gemini API client.
      model_name: Name of the model to use.
      initial_doc: The formatted input document.
      max_turn: Maximum number of search turns.
      search_step_min: Minimum target search steps.
      search_step_max: Maximum target search steps (0 for unspecified).

  Returns:
      A tuple of (question, answer, initial_doc, generator_response,
      prompt, search_step, answer_type).
  """
  answer_types = ["a number", "a date", "an entity"]
  answer_type = random.choice(answer_types)
  if search_step_max == 0:
    search_step = "multiple"
  else:
    search_step = random.choice(
        list(range(search_step_min, search_step_max + 1))
    )

  prompt = _PROMPT_MAP["with_search"].format(
      context=initial_doc,
      n_search_step=max_turn,
      answer_type=answer_type,
      target_search_step=search_step,
  )
  final_qa_pair, generator_response = utils.agentic_question_gen(
      client=client,
      model_name=model_name,
      prompt=prompt,
      max_turn=max_turn,
  )

  if final_qa_pair:
    question, answer = final_qa_pair
    print(">> Generated question, answer:", question, answer)
    return (
        question,
        answer,
        initial_doc,
        generator_response,
        prompt,
        search_step,
        answer_type,
    )
  else:
    print(">> No question generated.")
    return (
        None,
        None,
        initial_doc,
        generator_response,
        prompt,
        search_step,
        answer_type,
    )


def main():
  parser = argparse.ArgumentParser(
      description="Generate initial question-answer pairs"
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.0-flash",
      help="Model name for generation",
  )
  parser.add_argument("--n_sample", type=int, default=0)
  parser.add_argument(
      "--corpus_file",
      type=str,
      default="Search-R1/data/index/wiki-18.jsonl",
  )
  parser.add_argument("--save_sampled_corpus", action="store_true")
  parser.add_argument("--starting_idx", type=int, default=0)
  parser.add_argument(
      "--retriever_name",
      type=str,
      default="e5",
  )
  parser.add_argument(
      "--max_turn",
      type=int,
      default=10,
      help="Maximum search steps for the question generation agent",
  )
  parser.add_argument(
      "--max_num_workers",
      type=int,
      default=8,
      help="Maximum number of concurrent workers",
  )
  parser.add_argument(
      "--fixed_seed",
      action="store_true",
      help="Use a fixed seed for reproducibility",
  )
  parser.add_argument(
      "--initial_prompt",
      type=str,
      default="with_search",
  )
  parser.add_argument("--min_search_step", type=int, default=2)
  parser.add_argument("--max_search_step", type=int, default=5)
  args = parser.parse_args()

  project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
  location = os.environ.get("GOOGLE_CLOUD_LOCATION")
  if not project_id:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")
  if not location:
    raise ValueError("GOOGLE_CLOUD_LOCATION environment variable is not set")

  client = genai.Client(vertexai=True, project=project_id, location=location)

  curr_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

  if args.n_sample > 0:
    corpus = datasets.load_dataset(
        "json",
        data_files=args.corpus_file,
        split="train",
        num_proc=4,
    )
    print(">> loaded corpus file:", args.corpus_file)
    if args.fixed_seed:
      # pytype: disable=attribute-error
      sampled_docs = corpus.shuffle(seed=42).select(range(args.n_sample))
      # pytype: enable=attribute-error
    else:
      # pytype: disable=attribute-error
      sampled_docs = corpus.shuffle().select(range(args.n_sample))
      # pytype: enable=attribute-error
    sampled_docs_df = sampled_docs.to_pandas()
    if args.save_sampled_corpus:
      path = (
          "outputs/agentic_question_gen/"
          f"sampled_corpus_{args.n_sample}_{curr_timestamp}.jsonl"
      )
      sampled_docs_df.to_json(path, orient="records", lines=True)
      print(">> saved sampled corpus to:", path)
    corpus_name = f"wiki_{curr_timestamp}"
  else:
    corpus = pd.read_json(args.corpus_file, lines=True)
    print(">> loaded corpus file:", args.corpus_file)
    sampled_docs_df = corpus[args.starting_idx :]
    if "contents" not in sampled_docs_df.columns:
      sampled_docs_df["contents"] = sampled_docs_df["initial_doc"]
    corpus_name = args.corpus_file.split("/")[-1].replace(".jsonl", "")

  initial_docs = [
      construct_input_doc(row["contents"])
      for _, row in sampled_docs_df.iterrows()
  ]

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.max_num_workers
  ) as executor:
    futures = [
        executor.submit(
            question_gen_pipeline,
            client,
            args.model_name,
            doc,
            args.max_turn,
            args.min_search_step,
            args.max_search_step,
        )
        for doc in initial_docs
    ]
    all_responses = [
        future.result()
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(initial_docs),
        )
    ]

  rows = []
  for idx, response in enumerate(all_responses):
    question, answer, doc, gen_response, prompt, steps, atype = response
    if not question:
      print(f">> No question generated for sample {idx}")
      continue
    rows.append({
        "initial_doc": doc,
        "question": question,
        "answer": answer,
        "model_response": gen_response,
        "prompt": prompt,
        "search_steps": steps,
        "answer_type": atype,
    })

  output_df = pd.DataFrame(rows)
  output_file = (
      f"{corpus_name}_{args.model_name}_{args.retriever_name}"
      f"_with_search_{args.n_sample}_{args.max_turn}turn"
      f"_{args.min_search_step}to{args.max_search_step}_steps.jsonl"
  )
  output_path = f"outputs/agentic_question_gen/initial_questions/{output_file}"
  output_df.to_json(output_path, orient="records", lines=True)
  print(">> saved output to:", output_path)


if __name__ == "__main__":
  main()
