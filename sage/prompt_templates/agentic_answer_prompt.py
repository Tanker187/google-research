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

"""Prompt templates for agentic answer generation."""

EXAMPLE_ANSWER = "Beijing"

GEMINI_PROMPT = f"""Answer the given question by using a search engine. \
You will first reason about the question inside <think> and </think>, for instance, break down the question into multiple sub-questions that you will search for. \
You must call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
Try to formulate the search query in the form of a question. \
After receiving the information, you must reason about it inside <think> and </think> before issuing a new query or providing the final answer. \
Each of your reasoning step should be grounded in the retrieved information.  Do not use your own knowledge, but you can use commonsense knowledge or arithmetic knowledge. \
Do not use your own knowledge to write the query, the query should be based on the question and the retrieved documents. \
Do not infer the entities in the question, but you can use the entities in the retrieved documents to write the query. \
You can search as many times as your want. Try to break down the question for each search query and gather comprehensive information. \
If you have gathered enough information to answer the question, you can provide the answer to the query inside <answer> and </answer>, without detailed illustrations. \
Generate an answer based on the rerieved information, instead of your own knowledge. \
This is an example answer: <answer>{EXAMPLE_ANSWER}</answer>. Question:"""
