#!/bin/bash

MODELS="Qwen-Image"
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-HR, R-PR, R-GR, R-AR, R-CR, R-RR"
MLLM="Gemini_2_5_Flash"

# GPU mapping for different models
declare -A MLLM_GPU_MAP=(
    ["Qwen2_5_VL_72B"]=4
    ["Qwen3_VL_8B_Thinking"]=1
    ["Qwen3_VL_8B_Instruct"]=1
    ["Qwen3_VL_32B_Thinking"]=2
    ["Qwen3_VL_32B_Instruct"]=2
    ["Qwen3_VL_30B_A3B_Instruct"]=2
    ["Qwen3_VL_30B_A3B_Thinking"]=2
    ["Qwen3_VL_235B_A22B_Instruct"]=8
    ["Qwen3_VL_235B_A22B_Thinking"]=8
    ["Gemini_2_5_Flash"]=-1  # API-based model, no GPU specification needed
)

# Get the number of GPUs required for the selected MLLM
NUM_GPUS=${MLLM_GPU_MAP[$MLLM]}

# If model not found in map, use all available GPUs
if [ -z "$NUM_GPUS" ]; then
    echo "Warning: Model $MLLM not found in GPU map. Using all available GPUs."
    NUM_GPUS=-1
fi

# Set CUDA_VISIBLE_DEVICES based on the number of GPUs required
if [ "$NUM_GPUS" -eq -1 ]; then
    # Use all available GPUs (or API-based model)
    echo "Running $MLLM without GPU restriction"
elif [ "$NUM_GPUS" -gt 0 ]; then
    # Generate GPU list: 0,1,2,... up to NUM_GPUS-1
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Running $MLLM with $NUM_GPUS GPU(s): $GPU_LIST"
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
fi

# Start evaluation
python evaluate.py \
    --model "$MODELS" \
    --mllm "$MLLM" \
    --gen_eval_file "$DIMENSION"