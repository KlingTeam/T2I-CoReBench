#!/bin/bash

MODELS="Qwen-Image"
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-HR, R-PR, R-GR, R-AR, R-CR, R-RR"

# [HELP] Gemini_2_5_Flash, Qwen2_5_VL_72B, Qwen3_VL_32B_Instruct, Qwen3_VL_32B_Thinking, Qwen3_VL_30B_A3B_Instruct, Qwen3_VL_30B_A3B_Thinking, Qwen3_VL_235B_A22B_Instruct, Qwen3_VL_235B_A22B_Thinking
MLLM="Gemini_2_5_Flash"

# start evaluation
python evaluate.py \
    --model "$MODELS" \
    --mllm "$MLLM" \
    --gen_eval_file "$DIMENSION"