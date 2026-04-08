# POPO: Positive Only Policy Optimization with Implicit Negative Rewards

> **COLM 2026 under review** — A novel reinforcement learning algorithm that learns optimal policies using only positive examples, designed for Reinforcement Learning with Verifiable Rewards (RLVR) in mathematical reasoning.

---

## Overview

**Positive-Only Policy Optimization (POPO)** eliminates the need for explicit negative samples in RLVR. Instead of computing group-relative advantages over both correct and incorrect responses (GRPO), POPO trains exclusively on positive examples using three complementary mechanisms:

1. **Weighted NLL** over the positive set with self-normalized importance weights
2. **Similarity loss** between online and EMA-updated target policy representations
3. **Entropy regularization** to prevent mode collapse

```
L_POPO(θ, φ) = L_NLL(θ) + α · L_sim(θ, φ) + β · L_ent(θ)
```

### Why RLVR?

RLVR for mathematical reasoning provides the ideal setting: binary rewards (correct/incorrect), sparse signals (end-of-sequence), and deterministic verification (no reward model uncertainty). These properties make positive-only training both theoretically clean and practically effective.

| RLVR Property | POPO Alignment |
|---------------|----------------|
| Binary reward (0/1) | Exact positive/negative partition |
| Sparse reward (end only) | Sequence-level optimization |
| Deterministic verification | Clean positive set, no noise |
| No preference pairs needed | Positive-only training |

### Comparison with Related Methods

| Component | PPO | GRPO | DPO | **POPO** |
|-----------|-----|------|-----|----------|
| Critic / value network | Yes | No | No | **No** |
| Explicit negatives required | Yes | Yes | Yes | **No** |
| Preference pairs required | No | No | Yes | **No** |
| Advantage estimation | GAE | Group-relative | Implicit | **Binary (exact)** |
| Reference policy | Frozen | Frozen | Frozen | **EMA** |
| KL penalty | Yes | Yes | Implicit | **Replaced by similarity loss** |

## References
1. **GRPO** — Shao et al. (2024) "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
2. **Dr. GRPO** — Liu et al. (2025) "Understanding R1-Zero-Like Training: A Critical Perspective"
3. **RLVR** — Wen et al. (2025) "RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs"
4. **DPO** — Rafailov et al. (2023) "Direct Preference Optimization"
5. **PPO** — Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
6. **BYOL** — Grill et al. (2020) "Bootstrap Your Own Latent"
7. **DAPO** — Yu et al. (2025): Dynamic advantage normalization
8. **GSPO** — Qwen3: Group sequence policy optimization
9. **BAPO** — Balanced policy optimization with adaptive clipping

---

## Algorithm

### Core Equations

**Weighted Negative Log-Likelihood (policy learning over positives):**

$$L_\text{NLL}(\theta) = -\mathbb{E}_s\Big[\sum_{a \in S^+(s)} w_\theta(a|s) \log \pi_\theta(a|s)\Big]$$

**Similarity Loss (representation alignment via BYOL-style predictor):**

$$L_\text{sim}(\theta, \phi) = -\mathbb{E}_s\Big[\sum_{a \in S^+(s)} w_\theta(a|s) \cdot \cos\big(h_\phi(f_\theta(s,a)),\; \text{sg}(f_\xi(s,a) + \varepsilon)\big)\Big]$$

**Entropy Regularization (exploration maintenance):**

$$L_\text{ent}(\theta) = -\mathbb{E}_s\big[H(\pi_\theta(\cdot|s))\big]$$

**Weight Function (self-normalized over positive set):**

$$w_\theta(a|s) = \frac{\pi_\theta(a|s)}{Z^+(s)}, \quad Z^+(s) = \sum_{a' \in S^+(s)} \pi_\theta(a'|s)$$

**EMA Target Update:**

$$\xi \leftarrow \tau \cdot \xi + (1 - \tau) \cdot \theta$$

### Training Pipeline

```
Prompts → vLLM (G responses per prompt)
       → math_verify (binary reward 0/1)
       → Positive filter (reward > τ_r)
       → Single forward pass: logits + hidden states
       → L_NLL (weighted logps) + α·L_sim (predictor→target) + β·L_ent (entropy)
       → Gradient update θ, φ
       → EMA update ξ ← τξ + (1-τ)θ
```

### Theoretical Results

1. **Exact Advantage** — For binary RLVR rewards, POPO's implicit advantage equals the optimal advantage
2. **No Critic Required** — Eliminates value network, TD learning, and GAE
3. **Implicit Negatives** — Shared parameters redistribute probability from negative to positive actions
4. **Regret Bound** — O(√(T · |A| · log|A|))

### Hyperparameters

| Symbol | Name | Default | Range | Notes |
|--------|------|---------|-------|-------|
| α | Similarity weight | 0.1 | [0.05, 0.3] | Higher = stronger target anchoring |
| β | Entropy coefficient | 0.01 | [0.005, 0.05] | Higher = more exploration |
| τ | EMA momentum | 0.995 | [0.99, 0.999] | Higher = slower target evolution |
| G | Group size | 8 | [4, 16] | Responses per prompt |
| τ_r | Reward threshold | 0.0 | 0 for binary RLVR | Positive filtering cutoff |

---

## Project Structure

```
rlvr/
├── src/                               # Source code
│   ├── __init__.py
│   ├── utils.py                       # Config loading, experiment dirs, logging
│   ├── popo/                          # Core POPO algorithm
│   │   ├── config.py                  # POPOConfig(GRPOConfig)
│   │   ├── trainer.py                 # POPOTrainer(GRPOTrainer)
│   │   ├── loss.py                    # L_NLL + L_sim + L_ent + KL monitoring
│   │   ├── weights.py                 # Bounded importance weights over S⁺
│   │   ├── ema.py                     # EMA target policy manager
│   │   ├── predictor.py              # Predictor MLP h_φ (LayerNorm)
│   │   └── callbacks.py              # EMA update callback (DDP/FSDP safe)
│   ├── reward/                        # Reward computation
│   │   ├── math_verify_reward.py      # Binary reward via math_verify
│   │   └── extraction.py             # Answer extraction (boxed, ####, tags)
│   ├── data/                          # Data handling
│   │   ├── templates.py               # 6 prompt templates (R1, Qwen, Llama, etc.)
│   │   └── datasets.py               # HuggingFace dataset loading + formatting
│   └── evaluation/                    # Evaluation
│       ├── passk.py                   # Unbiased Pass@k estimator
│       └── evaluator.py              # vLLM-based Pass@k evaluation
├── scripts/                           # Runnable scripts
│   ├── train_grpo.py                  # GRPO baseline training
│   ├── train_popo.py                  # POPO training
│   ├── evaluate.py                    # Pass@k evaluation
│   └── shell/                         # Example launch scripts
│       ├── train_grpo_7b.sh
│       ├── train_grpo_1.5b.sh
│       ├── train_popo_7b.sh
│       └── eval_passk.sh
├── configs/                           # Modular YAML configs
│   ├── accelerate/{ddp,fsdp}.yaml
│   ├── models/{qwen_1.5b,qwen_7b}.yaml
│   ├── datasets/{gsm8k,math}.yaml
│   ├── trainer/default.yaml           # Base HuggingFace Trainer defaults
│   ├── grpo/default.yaml              # GRPO algorithm params
│   └── popo/default.yaml              # POPO algorithm params (primary file)
├── tests/                             # Unit tests (43 tests)
│   ├── test_predictor.py
│   ├── test_ema.py
│   ├── test_weights.py
│   ├── test_loss.py
│   └── test_reward.py
├── docs/progress/                     # Daily progress tracking
├── logs/                              # Experiment outputs (gitignored)
│   └── {experiment_name}/
│       ├── checkpoints/
│       ├── runs/                       # TensorBoard
│       ├── model_summary/
│       ├── train_config/
│       └── train_logs/
├── pyproject.toml
├── rlvr_environment.yaml              # Conda environment
```

---

## Getting Started

### Environment Setup

```bash
conda env create -f rlvr_environment.yaml
conda activate rlvr
```

## Frameworks

- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning
- [vLLM](https://github.com/vllm-project/vllm) — Fast inference

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| trl | 0.27.0 | Base GRPOTrainer |
| transformers | 4.57.6 | Model loading, Trainer |
| vllm | 0.11.2 | Fast response generation |
| math_verify | 0.9.0 | Mathematical answer verification |
| accelerate | 1.12.0 | Multi-GPU (DDP, FSDP) |
| torch | 2.7.1 | Framework |

### Full Training (Multi-GPU, vLLM Server)

```bash
# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model Qwen/Qwen2.5-Math-7B

# Terminal 2: POPO training with FSDP
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=configs/accelerate/fsdp.yaml scripts/train_popo.py \
    --config configs/popo/default.yaml \
    --model_name Qwen/Qwen2.5-Math-7B --max_steps 1500 \
    --prompt_template qwen_math --dataset_name agentica-org/DeepScaleR-Preview-Dataset --train_split train \
    --learning_rate 1e-7 \
    --per_device_train_batch_size 2 --num_generations 8 --gradient_accumulation_steps 8 \
    --alpha 0.1 --beta_entropy 0.01 --tau 0.999 --tau_r 0.0 --feature_noise_std 0.01 \
    --weight_mode softmax  \
    --predictor_hidden_dim 4096 --predictor_layers 2 \
    --max_completion_length 1024 --max_prompt_length 512 \
    --save_steps 500 --save_total_limit 1 \
    --gradient_checkpointing \
    --use_vllm --vllm_mode server \
    --output_dir logs/DSR_popo_Qwen2.5-Math-7B
```

### Evaluation on AIME 25 for example

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model_path logs/DSR_popo_Qwen2.5-Math-7B/checkpoints/final \
    --dataset_name MathArena/aime_2025 \
    --dataset_config none --split train \
    --template qwen_math \
    --n_samples 128 --k_values 8\
    --temperature 0.7 
```

## Monitoring

All POPO metrics are logged to TensorBoard:
```bash
tensorboard --logdir logs/{experiment_name}/runs
```
----
