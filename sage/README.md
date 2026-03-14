# SAGE

This is the repository for the paper [SAGE: Steerable Agentic Data Generation
for Deep Search with Execution Feedback](https://arxiv.org/pdf/2601.18202).

## Installation
For data generation:
```
conda create -n data_gen python=3.10
conda activate data_gen

# we recommend installing torch with conda for faiss-gpu (check the cuda version for your system)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi


## gemini API
pip install google-genai

```

## Set-up retrieval server
Follow the instruction from [Search-R1][search-r1-quick-start] repo to
set up the local retrieval server for E5 on Wikipedia and download the wikipedia
corpus `wiki-18.jsonl` which will be used by the data generator.

[search-r1-quick-start]: https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file#quick-start

## Google Cloud configuration
All scripts that call the Gemini API read your Google Cloud project and region
from environment variables:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
```

## Run initial generation
The results will be saved to `outputs/agentic_question_gen/initial_questions`.
```
python agentic_question_gen_pipeline/initial_question_gen.py \
    --model_name gemini-2.5-flash \
    --n_sample 1000 \
    --max_num_workers 8 \
    --max_turn 20 \
    --max_search_step 10
```

## Evaluate the generated question by prompting gemini-2.5-flash
Run best-of-4 inference with gemini-2.5-flash and evaluate with llm as a judge.
The inference result will be saved to `outputs/agentic_search/` and the llm as a
judge result will be saved to `outputs/gemini_llm_judge/`.

```
# set file name to $file
file=
n_rollout=4
max_turn=10
python prompt_gemini_agentic_search_batch.py    \
    --input_file_path outputs/agentic_question_gen/initial_questions/${file}.jsonl  \
    --question_column question    \
    --gold_answer_column answer    \
    --output_path_name ${file}  \
    --model_name gemini-2.5-flash \
    --n_rollout $n_rollout     --temperature 1 \
    --max_num_workers 8 \
    --max_turn $max_turn

python prompt_gemini_llm_as_a_judge.py    \
    --input_file_path outputs/agentic_search/${file}_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_rollout}_rollouts_0_thinking.jsonl   \
    --temperature 0.0     --gold_answer_column answer
```

## Generate feedbacks
The results will be saved to `outputs/agentic_question_gen/feedbacks`.
```
python agentic_question_gen_pipeline/feedback_gen.py \
    --input_file_path outputs/gemini_llm_judge/${file}_gemini-2.5-flash_e5_top3_10turn_new_4_rollouts_0_thinking_gemini-2.0-flash_0.0.jsonl \
    --model_name gemini-2.5-flash \
    --output_name ${file}
```

Then, evaluate the questions generated with feedback:
```
python prompt_gemini_agentic_search_batch.py    \
    --input_file_path outputs/agentic_question_gen/feedbacks/${file}_gemini-2.5-flash_0turn.jsonl  \
    --question_column question    \
    --gold_answer_column answer    \
    --output_path_name ${file}_1st_feedback  \
    --model_name gemini-2.5-flash     --n_rollout $n_sample     --temperature 1 \
    --max_num_workers 8 \
    --max_turn $max_turn

python prompt_gemini_llm_as_a_judge.py    \
    --input_file_path outputs/agentic_search/${file}_1st_feedback_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_sample}_rollouts_0_thinking.jsonl   \
    --temperature 0.0     --gold_answer_column answer
```

## Process generated data
Finally, run the below script to process the generated data. This will save the
data to `outputs/agentic_question_gen/processed_data`.

The script combines results from the initial round and any number of feedback
rounds. The initial round and intermediate feedback rounds use a strict filter
(`best_of_n == 1` and `pass_check`), while the final feedback round uses a
relaxed filter (`best_of_n == 1`). Results are deduplicated by question.

```
python process_generated_data.py    \
    --initial_round_file outputs/gemini_llm_judge/${file}_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_rollout}_rollouts_0_thinking_gemini-2.0-flash_0.0.jsonl   \
    --feedback_round_files outputs/gemini_llm_judge/${file}_1st_feedback_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_rollout}_rollouts_0_thinking_gemini-2.0-flash_0.0.jsonl \
    --output_file_name ${file}_processed \
    --min_search_step 2
```

To include multiple feedback rounds, pass additional files
to `--feedback_round_files`:

```
python process_generated_data.py    \
    --initial_round_file outputs/gemini_llm_judge/${file}_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_rollout}_rollouts_0_thinking_gemini-2.0-flash_0.0.jsonl   \
    --feedback_round_files \
        outputs/gemini_llm_judge/${file}_1st_feedback_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_rollout}_rollouts_0_thinking_gemini-2.0-flash_0.0.jsonl \
        outputs/gemini_llm_judge/${file}_2nd_feedback_gemini-2.5-flash_e5_top3_${max_turn}turn_new_${n_rollout}_rollouts_0_thinking_gemini-2.0-flash_0.0.jsonl \
    --output_file_name ${file}_processed \
    --min_search_step 2
```

| Argument | Required | Description |
|---|---|---|
| `--initial_round_file` | yes | LLM judge results for the initial generation round |
| `--feedback_round_files` | no | LLM judge results for each feedback round, in order (one or more files) |
| `--output_file_name` | yes | Output file base name (without `.jsonl` extension) |
| `--min_search_step` | no | Filter out questions requiring fewer than N search steps (default: 0) |

## Training with the generated data (RL)

### Installation

Follow the Search-R1 [guide](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file#installation) to set up environment.

Note: if encountering flash-attention related error, try installing with below
command:

```
pip3 install flash-attn==2.7.3 --no-build-isolation
```

### Prepare the data

First, convert the generated data into parquet for training. If you want to
combine multiple files, you will need to separately pre-process them into a
single file first. Then run:

```
python create_training_data.py \
    --input_file outputs/agentic_question_gen/initial_questions/${file}.jsonl \
    --dataset_name  $dataset_name
```

### Then run this script

Update the $train_file_name in the script
```
bash run.sh
```

### Generated data

We release the generated data based on wikipedia [here](https://huggingface.co/datasets/fangyuan/sage-data).

It consists of:

- 20k data generated by the initial data generator
- 20k data generated with one round of feedback
- 20k data generated with two rounds of feedback
- 20k data generated with three rounds of feedback
- the in-domain testing data requiring 2-7 search steps (300 data each)

### Citation
If you find our work helpful, please cite us as
```
@inproceedings{
xu2026steerable,
title={Steerable Agentic Data Generation for Deep Search with Execution Feedback},
author={Fangyuan Xu and Rujun Han and Yanfei Chen and Zifeng Wang and I-Hung Hsu and Jun Yan and Vishy Tirumalashetty and Eunsol Choi and Tomas Pfister and Chen-Yu Lee},
booktitle={19th Conference of the European Chapter of the Association for Computational Linguistics},
year={2026},
url={https://arxiv.org/pdf/2601.18202}
}
```
