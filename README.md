# POPO: Positive-Only Policy Optimization for RLVR

> **NeurIPS 2026 Submission** — A novel reinforcement learning algorithm that learns optimal policies using only positive examples, designed for Reinforcement Learning with Verifiable Rewards (RLVR) in mathematical reasoning.

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
├── test_environment/                  # Self-contained GRPO prototype
│   ├── train_grpo_v2.py               # Working GRPO training (canonical source)
│   └── ...
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
├── POPO_IMPLEMENTATION_GUIDE.md       # Detailed implementation reference
└── POPO_QUICK_REFERENCE.md            # One-page algorithm card
```

---

## Getting Started

### Environment Setup

```bash
conda env create -f rlvr_environment.yaml
conda activate rlvr
```

### Quick Test (Single GPU)

```bash
# POPO debug run (1.5B, 10 steps)
python scripts/train_popo.py \
    --config configs/popo/default.yaml \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --dataset_name openai/gsm8k \
    --max_steps 10 \
    --per_device_train_batch_size 1 \
    --num_generations 4 \
    --report_to none

# GRPO baseline (same setup)
python scripts/train_grpo.py \
    --config configs/grpo/default.yaml \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --dataset_name openai/gsm8k \
    --max_steps 10 \
    --report_to none
```

### Full Training (Multi-GPU, vLLM Server)

```bash
# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model Qwen/Qwen2.5-Math-7B

# Terminal 2: POPO training with FSDP
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file=configs/accelerate/fsdp.yaml \
    scripts/train_popo.py \
    --config configs/popo/default.yaml \
    --model_name Qwen/Qwen2.5-Math-7B \
    --dataset_name openai/gsm8k \
    --use_vllm --vllm_mode server \
    --output_dir logs/popo_7b_gsm8k
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model_path logs/popo_7b_gsm8k/checkpoints/final \
    --dataset_name openai/gsm8k \
    --n_samples 128 \
    --k_values 1 8 64 128 \
    --eval_greedy
```

### Unit Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

The project uses **argparse + YAML defaults**: YAML config files set parser defaults, CLI arguments override them.

```bash
# YAML sets defaults, CLI overrides specific values
python scripts/train_popo.py \
    --config configs/popo/default.yaml \    # loads all defaults
    --alpha 0.2 \                           # override similarity weight
    --tau 0.999                             # override EMA momentum
```

Primary config file for POPO experiments: `configs/popo/default.yaml`

---

## Monitoring

All POPO metrics are logged to TensorBoard:

| Metric | Key | Healthy Range | Action if Unhealthy |
|--------|-----|--------------|---------------------|
| Reward mean | `reward` | ↑ over time | Check data/reward function |
| Mean entropy | `entropy` | [1.0, 3.0] | Increase β if < 0.5 |
| Positive ratio | `popo/positive_ratio` | > 0.1 | Use easier data if < 0.05 |
| NLL loss | `popo/nll_loss` | ↓ steadily | Check learning rate |
| Similarity loss | `popo/sim_loss` | ↓ slowly | Adjust α or τ |
| KL divergence | `popo/kl_divergence` | Stable | Decrease τ if spiking |

```bash
tensorboard --logdir logs/{experiment_name}/runs
```

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| trl | 0.27.0 | Base GRPOTrainer |
| transformers | 4.57.6 | Model loading, Trainer |
| vllm | 0.11.2 | Fast response generation |
| math_verify | 0.9.0 | Mathematical answer verification |
| accelerate | 1.12.0 | Multi-GPU (DDP, FSDP) |
| peft | 0.18.1 | LoRA fine-tuning |
| torch | 2.7.1 | Framework |

Full environment: `rlvr_environment.yaml`

---

## Research Roadmap

### Phase 1: Foundation (Current)
- [x] GRPO baseline infrastructure (`test_environment/`)
- [x] Professional project scaffolding
- [x] Core POPO implementation (7 modules)
- [x] Unit tests (43 passing)
- [ ] End-to-end GPU validation

### Phase 2: Baseline Experiments
- [ ] Qwen2.5-Math-1.5B on GSM8K (POPO vs GRPO)
- [ ] Hyperparameter sensitivity: α, β, τ
- [ ] Qwen2.5-Math-7B full training

### Phase 3: Ablation Studies
- [ ] Component ablations (w/o sim, w/o ent, w/o EMA)
- [ ] Architecture: mean pooling vs last token, predictor depth
- [ ] Training: group size G, LoRA rank

### Phase 4: Scaling & Analysis
- [ ] Qwen2.5-Math-32B (if compute available)
- [ ] Training dynamics (gradient flow, implicit negatives)
- [ ] Efficiency analysis (time, memory, samples)

### Phase 5: Paper
- [ ] Comprehensive evaluation (MATH-500, AMC23, AIME24/25)
- [ ] Pass@K and CoT-Pass@K analysis
- [ ] NeurIPS 2026 submission

---

## Benchmarks

| Benchmark | Domain | Size | Metric |
|-----------|--------|------|--------|
| GSM8K | Grade school math | 8K train / 1.3K test | Accuracy |
| MATH | Competition math | 7.5K train / 5K test | Accuracy |
| MATH-500 | MATH subset | 500 test | Accuracy |
| AMC23 | AMC competition | ~25 problems | Accuracy |
| AIME24/25 | AIME competition | 30 problems | Accuracy |

---

## References

### Core Papers

1. **GRPO** — Shao et al. (2024) "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
2. **Dr. GRPO** — Liu et al. (2025) "Understanding R1-Zero-Like Training: A Critical Perspective"
3. **RLVR** — Wen et al. (2025) "RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs"
4. **DPO** — Rafailov et al. (2023) "Direct Preference Optimization"
5. **PPO** — Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
6. **BYOL** — Grill et al. (2020) "Bootstrap Your Own Latent"

### Related Methods

- **DAPO** — Yu et al. (2025): Dynamic advantage normalization
- **GSPO** — Qwen3: Group sequence policy optimization
- **BAPO** — Balanced policy optimization with adaptive clipping

### Frameworks

- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning
- [vLLM](https://github.com/vllm-project/vllm) — Fast inference
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Open RL for LLMs

---

**Target Venue:** NeurIPS 2026  
**Last Updated:** February 2026
