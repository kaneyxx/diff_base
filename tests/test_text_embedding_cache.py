"""Tests for the text embedding cache wired into KontextTrainerMixin.

Scenarios:
1. Same caption 4 times → encode_text called exactly once.
2. Mixed captions [A, A, B, A] → encode_text called twice (A + B each once).
3. CPU smoke: second call returns same tensors without invoking encode_text again.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from src.training.kontext_trainer import KontextTrainerMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 32


def _make_mixin_with_cache(tmp_path: str) -> KontextTrainerMixin:
    """Build a KontextTrainerMixin with text cache enabled and a fake model."""
    config = OmegaConf.create({
        "training": {"cache_text_embeddings": True},
        "data": {"cache_dir": tmp_path},
    })
    mixin = KontextTrainerMixin()
    mixin.config = config
    mixin._init_text_cache(config)
    return mixin


def _make_fake_model(call_counter: list[int]) -> MagicMock:
    """Fake model whose encode_text increments a counter and returns fixed tensors."""
    model = MagicMock()

    def encode_text(captions: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        call_counter[0] += 1
        B = len(captions)  # noqa: N806
        return {
            "prompt_embeds": torch.zeros(B, 77, HIDDEN),
            "pooled_prompt_embeds": torch.zeros(B, HIDDEN),
        }

    model.encode_text = encode_text
    return model


# ---------------------------------------------------------------------------
# Scenario 1: same caption 4 times → encode_text called exactly once
# ---------------------------------------------------------------------------

class TestSameCaptionRepeat:

    def test_encode_called_once_for_same_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mixin = _make_mixin_with_cache(tmp)
            counter = [0]
            model = _make_fake_model(counter)
            device = torch.device("cpu")

            caption = "Generate CD45 biomarker image from H&E"
            for _ in range(4):
                out = mixin._get_cached_text_output([caption], device, model)

            assert counter[0] == 1, (
                f"encode_text should be called exactly once for repeated captions, "
                f"but was called {counter[0]} times."
            )

    def test_output_keys_match_training_step_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mixin = _make_mixin_with_cache(tmp)
            counter = [0]
            model = _make_fake_model(counter)
            device = torch.device("cpu")

            out = mixin._get_cached_text_output(["test caption"], device, model)

            assert "prompt_embeds" in out, "Missing 'prompt_embeds' key in output."
            assert "pooled_prompt_embeds" in out, "Missing 'pooled_prompt_embeds' key in output."


# ---------------------------------------------------------------------------
# Scenario 2: mixed captions [A, A, B, A] → encode_text called exactly twice
# ---------------------------------------------------------------------------

class TestMixedCaptions:

    def test_encode_called_once_per_unique_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mixin = _make_mixin_with_cache(tmp)
            counter = [0]
            model = _make_fake_model(counter)
            device = torch.device("cpu")

            caption_a = "Generate CD45 biomarker image from H&E"
            caption_b = "Generate CD3 biomarker image from H&E"

            # First call encodes A (miss)
            mixin._get_cached_text_output([caption_a], device, model)
            # Second call hits cache for A, encodes B (miss)
            mixin._get_cached_text_output([caption_a, caption_b], device, model)
            # Third call: A and B both cached
            mixin._get_cached_text_output([caption_a, caption_a, caption_b], device, model)

            assert counter[0] == 2, (
                f"encode_text should be called exactly twice (once per unique caption), "
                f"but was called {counter[0]} times."
            )

    def test_batch_output_shape_correct_for_mixed(self) -> None:
        """Output tensors must have batch dim = number of captions in call."""
        with tempfile.TemporaryDirectory() as tmp:
            mixin = _make_mixin_with_cache(tmp)
            counter = [0]
            model = _make_fake_model(counter)
            device = torch.device("cpu")

            caption_a = "caption A"
            caption_b = "caption B"

            out = mixin._get_cached_text_output(
                [caption_a, caption_a, caption_b, caption_a], device, model
            )

            assert out["prompt_embeds"].shape[0] == 4, (
                f"Expected batch dim 4, got {out['prompt_embeds'].shape[0]}."
            )
            assert out["pooled_prompt_embeds"].shape[0] == 4


# ---------------------------------------------------------------------------
# Scenario 3: CPU smoke — second call returns same tensors without invoking model
# ---------------------------------------------------------------------------

class TestCacheSmokeSecondCall:

    def test_second_call_no_encode(self) -> None:
        """After first call, second call must return without invoking encode_text."""
        with tempfile.TemporaryDirectory() as tmp:
            mixin = _make_mixin_with_cache(tmp)
            counter = [0]
            model = _make_fake_model(counter)
            device = torch.device("cpu")

            caption = "test caption for smoke"

            out1 = mixin._get_cached_text_output([caption], device, model)
            out2 = mixin._get_cached_text_output([caption], device, model)

            assert counter[0] == 1, (
                f"encode_text must not be called on second hit; counter = {counter[0]}."
            )

            # Shapes must match
            assert out1["prompt_embeds"].shape == out2["prompt_embeds"].shape
            assert out1["pooled_prompt_embeds"].shape == out2["pooled_prompt_embeds"].shape

    def test_cache_disabled_by_default(self) -> None:
        """When cache_text_embeddings is false (default), _text_cache must be None."""
        config = OmegaConf.create({
            "training": {},  # no cache_text_embeddings key
            "data": {},
        })
        mixin = KontextTrainerMixin()
        mixin._init_text_cache(config)
        assert mixin._text_cache is None, (
            "_text_cache should be None when cache_text_embeddings is not set."
        )

    def test_cache_disabled_explicit_false(self) -> None:
        """Explicit cache_text_embeddings: false → _text_cache is None."""
        config = OmegaConf.create({
            "training": {"cache_text_embeddings": False},
            "data": {},
        })
        mixin = KontextTrainerMixin()
        mixin._init_text_cache(config)
        assert mixin._text_cache is None
