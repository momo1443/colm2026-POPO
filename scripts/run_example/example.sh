# Evaluate a trained POPO model
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
    --model_path logs/popo_qwen7b_gsm8k/checkpoints/final \
    --dataset_name openai/gsm8k \
    --template qwen_math \
    --n_samples 128 --k_values 1 8 64 128 \
    --temperature 0.7 --eval_greedy

# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model Qwen/Qwen2.5-Math-7B

# Terminal 2: Run POPO training
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file=configs/accelerate/fsdp.yaml \
    scripts/train_popo.py \
    --config configs/popo/default.yaml \
    --model_name Qwen/Qwen2.5-Math-7B \
    --dataset_name openai/gsm8k \
    --prompt_template qwen_math \
    --max_steps 1000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --max_completion_length 512 \
    --alpha 0.1 --beta_entropy 0.01 --tau 0.995 \
    --predictor_hidden_dim 1024 \
    --use_vllm --vllm_mode server \
    --do_eval --eval_steps 25 \
    --save_steps 500 \
    --output_dir logs/popo_qwen7b_gsm8k


#==============================================================================


# Evaluate a trained GRPO model
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py \
    --model_path logs/grpo_qwen7b_gsm8k/checkpoints/final \
    --dataset_name openai/gsm8k \
    --template qwen_math \
    --n_samples 128 --k_values 1 8 64 128 \
    --temperature 0.7 --eval_greedy

# =============================================================================
# GRPO Training: Qwen2.5-Math-1.5B on GSM8K
# =============================================================================
# Terminal 1: Start vLLM server
# CUDA_VISIBLE_DEVICES=4,5 trl vllm-serve --model Qwen/Qwen2.5-Math-1.5B --tensor_parallel_size 2

# Terminal 2: Run training
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file=configs/accelerate/fsdp.yaml \
    scripts/train_grpo.py \
    --config configs/grpo/default.yaml \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --dataset_name openai/gsm8k \
    --prompt_template qwen_math \
    --max_steps 2000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --max_completion_length 512 \
    --use_vllm --vllm_mode server \
    --do_eval --eval_steps 25 \
    --save_steps 1000 \
    --output_dir logs/grpo_qwen1.5b_gsm8k




