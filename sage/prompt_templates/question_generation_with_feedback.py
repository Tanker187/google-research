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

"""Prompt templates for question generation with feedback in the SAGE framework."""

# Initial generation

QUESTION_GENERATION_WITH_SEARCH = """Your task is to generate a complicated question that will require a search agent {target_search_step} search steps to answer by gathering information using a search engine.
You will first reason about the initial document and plan for gathering comprehensive information inside <think> and </think>. \
You will then call a search engine by <search> query </search> and it will return the top searched results between <information> and </information> to collect information.
You must conduct reasoning inside <think> and </think> first every time you get new information.
You will call the search engine for {n_search_step} steps.
After {n_search_step} searches, you must provide the question inside <question> and </question>, the answer inside <answer> and </answer>, and the answering step inside <answering steps> and </answering steps>. \
You can use your own knowledge to construct the search query, but the final answer and each of the answering step must be supported by the information you gathered from the search engine. \
The question should be understandable standalone as the agent will use the question to search for information without access to the initial document. \
An example question: How much did the film in which Jake Gyllenhaal played his second lead role gross in its initial run at the box office? \
Avoid How and Why question. \
The answer should be {answer_type} and short. \
Make sure the answer is correct and **unique** for the question generated. \

Initial document:
{context}
"""

CORRECTNESS_FEEDBACK_PROMPT_NO_SEARCH = r"""You will be given an output from a question generator agent, which generates a complicated question, answer pair; as well as the output from a search agent, which attempts to solve the question generated in a fixed number of turns.

The answer from the search agent is not the same as the data generator agent. You task is to examine their traces and output the correct question, answer pair based on their retrieved documents. You can update either the question, the answer or both.

You will first reason about why is there a discrepancy between the search agent's answer and the data generator's answer. Output your reasoning trace inside <reason> and </reason>.
You will then reason about how to update the question answer pair to make sure it is correct and requires the agent {target_step} search step to answer. A search step is defined as a call to the search tool. Output your reasoning trace inside <think> and </think>.
For factual information, you should ONLY rely on the context provided for the data generator agent and the documents retrieved by both the data generator and search agent (inside <information> and </information>).

If you find it non-trivial to update just the question and answer, you can generate a new question answer pair ONLY based on the retrieved documents.

The updated question should require the search agent at least {target_step} search steps to answer. However, the answer should be short, such as an entity, a date or a number. The question should be understandable standalone, as the search agent will solve the question without access to the documents (they will need to search for them).

When you are ready to provide the new question, answer pair, you can provide the question inside <question> and </question>, the answer inside <answer> and </answer>, and the search step inside <search steps> and </search steps>.
For each search step, output the exact search question; the sub-answer to the search question; and the retrieved document from the search agent and data generator agent's output that supports the sub-answer.
Make sure each step is absolutely needed to answer the question and there is no short cut. Tip: use retrieved document from different steps so avoid two sub-queries being solved by one search query.

# Data generator agent
Prompt:  {data_generator_agent_prompt}
Agent's output:  {data_generator_agent_response}

# Search agent
Prompt: {search_agent_prompt}
Agent's output: {search_agent_response}

# Your output
"""

DIFFICULTY_FEEDBACK_PROMPT_NO_SEARCH = r"""You will be given an output from a question generator agent, which generates a complicated question, answer pair to be solved by a search agent for at least {target_step} **search** steps; as well as the output from a search agent, which attempts to solve the question generated. The search agent is able to solve the question in less than {target_step} search steps. Your task is to update the question so that it requires the search agent more steps to solve.

You will first reason about why the search agent is able to solve the question in fewer steps. Output your reasoning trace inside <reason> and </reason>.
You will then reason about how to update the question so that it will require more search steps.
For factual information, you should ONLY rely on the context provided for the data generator agent and the documents retrieved by both the data generator and search agent (inside <information> and </information>), without relying on other information not in the retrieved context.
Output your reasoning trace inside <think> and </think>.
If you find it non-trivial to update the plan, you can generate a new question answer pair ONLY based on the retrieved documents.

The updated question should require the search agent at least {target_step} search steps to answer. Note that some of the answering steps do not involve search and thus do not count. However, the answer should be short, such as an entity, a date or a number. The question should be understandable standalone, as the agent will solve the question without access to the documents (they will need to search for them).

When you are ready to provide the new question, answer pair, you can provide the question inside <question> and </question>, the answer inside <answer> and </answer>, and the search step inside <search steps> and </search steps>. For each search step, output the exact search question; the sub-answer to the search question; and the retrieved document from the search agent and data generator agent's output that supports the sub-answer. Make sure each step is absolutely needed to answer the question and there is no short cut. Tip: use retrieved document from different steps so avoid two sub-queries being solved by one search query.

# Data generator agent
Prompt:  {data_generator_agent_prompt}
Agent's output:  {data_generator_agent_response}

# Search agent
Prompt: {search_agent_prompt}
Agent's output: {search_agent_response}

# Your output
"""
