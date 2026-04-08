"""Tests for the refactored _compute_loss memory-efficient utilities.

Covers:
1. _chunked_entropy_from_logits correctness, gradient flow, dtype handling
2. Forward hook on model.norm captures same hidden states as output_hidden_states
3. End-to-end gradient flow through hook + selective_log_softmax + chunked entropy
"""

import pytest
import torch
from transformers import AutoConfig, Qwen2ForCausalLM
from trl.trainer.utils import selective_log_softmax

from src.popo.trainer import _chunked_entropy_from_logits, _masked_mean_pool
from src.popo.predictor import PredictorMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _naive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Reference entropy: -(softmax(x) * log_softmax(x)).sum(-1), in fp32."""
    logits_f = logits.float()
    log_p = logits_f.log_softmax(dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def _make_tiny_qwen2(hidden_size=64, num_layers=2, vocab_size=1000):
    config = AutoConfig.for_model(
        "qwen2",
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(config)


# ===================================================================
# 1. _chunked_entropy_from_logits
# ===================================================================

class TestChunkedEntropy:

    @pytest.mark.parametrize("shape", [
        (2, 10, 100),
        (1, 1, 50000),
        (4, 8, 256),
    ])
    def test_matches_naive(self, shape):
        torch.manual_seed(0)
        logits = torch.randn(*shape)
        expected = _naive_entropy(logits)
        actual = _chunked_entropy_from_logits(logits)
        assert torch.allclose(actual, expected, atol=1e-5), (
            f"Max diff: {(actual - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("chunk_size", [1, 7, 128, 10000])
    def test_chunk_size_invariance(self, chunk_size):
        torch.manual_seed(1)
        logits = torch.randn(3, 12, 200)
        ref = _chunked_entropy_from_logits(logits, chunk_size=10000)
        out = _chunked_entropy_from_logits(logits, chunk_size=chunk_size)
        assert torch.allclose(out, ref, atol=1e-5), (
            f"chunk_size={chunk_size}: max diff {(out - ref).abs().max().item()}"
        )

    def test_gradient_flow(self):
        logits = torch.randn(2, 8, 100, requires_grad=True)
        ent = _chunked_entropy_from_logits(logits)
        ent.sum().backward()
        assert logits.grad is not None, "No gradient on logits"
        assert not torch.isnan(logits.grad).any(), "NaN in gradient"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_input(self, dtype):
        """Function should accept fp16/bf16 and produce fp32 output."""
        torch.manual_seed(2)
        logits_fp32 = torch.randn(2, 8, 200)
        logits_lp = logits_fp32.to(dtype)
        out = _chunked_entropy_from_logits(logits_lp)
        assert out.dtype == torch.float32, f"Expected fp32, got {out.dtype}"
        ref = _chunked_entropy_from_logits(logits_fp32)
        assert torch.allclose(out, ref, atol=0.05), (
            f"Low-precision max diff: {(out - ref).abs().max().item()}"
        )

    def test_output_shape(self):
        logits = torch.randn(3, 7, 50)
        out = _chunked_entropy_from_logits(logits)
        assert out.shape == (3, 7)

    def test_entropy_nonnegative(self):
        logits = torch.randn(4, 16, 100)
        ent = _chunked_entropy_from_logits(logits)
        assert (ent >= -1e-6).all(), f"Negative entropy: min={ent.min().item()}"

    def test_uniform_distribution_entropy(self):
        """Uniform logits -> entropy = log(V)."""
        V = 64
        logits = torch.zeros(1, 1, V)
        ent = _chunked_entropy_from_logits(logits)
        expected = torch.tensor(V, dtype=torch.float32).log()
        assert torch.allclose(ent.squeeze(), expected, atol=1e-5)

    def test_peaked_distribution_low_entropy(self):
        """One-hot-like logits -> entropy near 0."""
        V = 100
        logits = torch.full((1, 1, V), -100.0)
        logits[0, 0, 0] = 100.0
        ent = _chunked_entropy_from_logits(logits)
        assert ent.item() < 0.01


# ===================================================================
# 2. Forward hook captures correct hidden states
# ===================================================================

class TestForwardHook:

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return _make_tiny_qwen2()

    def test_hook_matches_output_hidden_states(self, model):
        """Hook on model.norm should capture same tensor as hidden_states[-1]."""
        model.eval()
        torch.manual_seed(0)
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out_full = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            ref_hidden = out_full.hidden_states[-1]

        captured = {}
        handle = model.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden", o)
        )
        try:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()

        assert "hidden" in captured, "Hook did not fire"
        assert torch.allclose(captured["hidden"], ref_hidden, atol=1e-5), (
            f"Max diff: {(captured['hidden'] - ref_hidden).abs().max().item()}"
        )

    def test_hook_shape(self, model):
        B, T = 3, 10
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones_like(input_ids)

        captured = {}
        handle = model.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden", o)
        )
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()

        assert captured["hidden"].shape == (B, T, model.config.hidden_size)

    def test_hook_cleanup_on_exception(self, model):
        """Hook handle must be removed even if forward raises."""
        handle = model.model.norm.register_forward_hook(
            lambda m, i, o: None
        )
        hooks_before = len(model.model.norm._forward_hooks)
        handle.remove()
        hooks_after = len(model.model.norm._forward_hooks)
        assert hooks_after == hooks_before - 1

    def test_hook_with_logits_to_keep(self, model):
        """Hook captures full-sequence hidden states even with logits_to_keep."""
        B, prompt_len, comp_len = 2, 8, 8
        T = prompt_len + comp_len
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones_like(input_ids)

        captured = {}
        handle = model.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden", o)
        )
        try:
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=comp_len + 1,
                )
        finally:
            handle.remove()

        assert captured["hidden"].shape == (B, T, model.config.hidden_size), (
            "Hook should capture all T positions even with logits_to_keep"
        )
        assert out.logits.shape[1] == comp_len + 1, (
            "Logits should be trimmed to logits_to_keep"
        )


# ===================================================================
# 3. Gradient flow through full pipeline
# ===================================================================

class TestPipelineGradientFlow:

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        model = _make_tiny_qwen2(hidden_size=64, num_layers=2, vocab_size=1000)
        model.train()
        predictor = PredictorMLP(
            input_dim=64, hidden_dim=32, output_dim=64, num_layers=2
        )
        return model, predictor

    def test_gradients_reach_model_parameters(self, setup):
        """Full pipeline: hook -> logps -> entropy -> predictor -> loss -> backward."""
        model, predictor = setup
        B, prompt_len, comp_len = 2, 6, 10
        input_ids = torch.randint(0, 1000, (B, prompt_len + comp_len))
        attention_mask = torch.ones_like(input_ids)
        completion_ids = input_ids[:, prompt_len:]
        completion_mask = torch.ones(B, comp_len)

        captured = {}
        handle = model.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden", o)
        )
        try:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        finally:
            handle.remove()

        logits = logits[:, :-1, :][:, -comp_len:, :]

        per_token_logps = selective_log_softmax(logits, completion_ids)
        entropies = _chunked_entropy_from_logits(logits)

        online_hidden = captured["hidden"][:, prompt_len:, :]
        online_features = _masked_mean_pool(online_hidden, completion_mask)
        predicted = predictor(online_features)

        nll = -(per_token_logps * completion_mask).sum() / completion_mask.sum()
        ent = -(entropies * completion_mask).sum() / completion_mask.sum()
        sim = -torch.nn.functional.cosine_similarity(
            predicted, predicted.detach(), dim=-1
        ).mean()
        loss = nll + 0.01 * ent + 0.1 * sim

        loss.backward()

        params_with_grad = 0
        params_total = 0
        for name, p in model.named_parameters():
            params_total += 1
            if p.grad is not None:
                params_with_grad += 1
                assert not torch.isnan(p.grad).any(), f"NaN grad: {name}"
        assert params_with_grad > 0, "No model parameters received gradients"

        for name, p in predictor.named_parameters():
            assert p.grad is not None, f"Predictor param {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"NaN grad in predictor: {name}"

    def test_hook_hidden_receives_gradient(self, setup):
        """Hidden states captured by hook should be part of the computation graph."""
        model, _ = setup
        B, T = 2, 12
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones_like(input_ids)

        captured = {}
        handle = model.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden", o)
        )
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()

        loss = captured["hidden"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Backprop through hooked hidden states should reach model params"

    def test_entropy_gradient_nonzero(self, setup):
        """Entropy branch should contribute non-zero gradients."""
        model, _ = setup
        B, T = 2, 8
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones_like(input_ids)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1, :]

        entropies = _chunked_entropy_from_logits(logits)
        loss = entropies.mean()
        loss.backward()

        lm_head_grad = model.lm_head.weight.grad
        assert lm_head_grad is not None, "lm_head should receive entropy gradient"
        assert lm_head_grad.abs().sum() > 0, "Entropy gradient should be non-zero"


# ===================================================================
# 4. _masked_mean_pool
# ===================================================================

class TestMaskedMeanPool:

    def test_basic_shape(self):
        hidden = torch.randn(3, 8, 64)
        mask = torch.ones(3, 8)
        out = _masked_mean_pool(hidden, mask)
        assert out.shape == (3, 64)

    def test_all_zero_mask(self):
        """All-zero mask should not produce NaN (clamped denominator)."""
        hidden = torch.randn(2, 4, 32)
        mask = torch.zeros(2, 4)
        out = _masked_mean_pool(hidden, mask)
        assert not torch.isnan(out).any(), "NaN with all-zero mask"

    def test_partial_mask(self):
        """Only masked positions should contribute."""
        hidden = torch.zeros(1, 4, 2)
        hidden[0, 0, :] = 10.0
        hidden[0, 1, :] = 20.0
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        out = _masked_mean_pool(hidden, mask)
        assert torch.allclose(out, torch.tensor([[10.0, 10.0]]))

    def test_batch_size_one(self):
        hidden = torch.randn(1, 5, 16)
        mask = torch.ones(1, 5)
        out = _masked_mean_pool(hidden, mask)
        assert out.shape == (1, 16)
