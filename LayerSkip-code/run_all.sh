#!/bin/bash

models=(
  # "facebook/layerskip-llama2-7B"
  # "facebook/layerskip-llama2-13B"
  # "facebook/layerskip-llama2-70B"
  # "facebook/layerskip-codellama-7B"
  # "facebook/layerskip-llama3-8B"
  # "facebook/layerskip-llama3.2-1B"
  "facebook/layerskip-codellama-34B"
)

for model in "${models[@]}"; do
#   echo "Running benchmark for $model"
#   torchrun benchmark.py \
#     --model "$model" \
#     --dataset cnn_dm_summarization \
#     --num_samples 100 \
#     --generation_strategy self_speculative \
#     --exit_layer 8 \
#     --num_speculations 6 \
#     --output_dir "./logs/${model//\//_}"

  echo "Running eval for $model"
  torchrun eval.py \
    --model "$model" \
    --tasks gsm8k \
    --limit 1 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir "./logs/${model//\//_}"
done
