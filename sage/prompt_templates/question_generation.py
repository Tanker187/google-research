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

"""Prompt templates for question generation and evaluation."""

LLM_AS_A_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.



[question]: {question}



[response]: {model_answer}



Your judgement must be in the format and criteria specified below:



extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.



[correct_answer]: {gold_answer}



reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Focus on recall, i.e. if the extracted_final_answer covers all the points of the [correct_answer]. It is ok if it provides more details. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match. Ignore capitalization.



correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.





confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


LLM_AS_A_JUDGE_PROMPT_LIST = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer_list] below. Each answer in the [correct_answer_list] is separated by a comma.



[question]: {question}



[response]: {model_answer}



Your judgement must be in the format and criteria specified below:



extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.



[correct_answer]: {gold_answer}



reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer_list], focusing only on if there are meaningful differences between answer in the [correct_answer_list] and the extracted_final_answer. Focus on recall, i.e. if the extracted_final_answer covers all the points in the answer in the [correct_answer_list]. It is ok if it provides more details.

It is also ok if the extracted_final_answer misses minor point from the correct_answer, as long as it is evident that they are referring to the same thing. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer_list], focus only on whether the answers match. Ignore capitalization.



correct: Answer 'yes' if extracted_final_answer matches any of the answers in [correct_answer_list] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.



confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""
