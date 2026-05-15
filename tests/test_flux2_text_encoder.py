"""FLUX.2 text encoder offline fallback tests (US-4).

Tests that:
1. HF_HUB_OFFLINE=1 causes local_files_only=True and raises a clear error
   when the model is not cached locally.
2. Without HF_HUB_OFFLINE, the standard (online) path is taken (local_files_only
   is NOT forced True).

No real HuggingFace downloads are performed; from_pretrained is monkeypatched.
"""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.models.flux.v2.text_encoder import Flux2TextEncoders

# ---------------------------------------------------------------------------
# Helper: build a minimal Flux2TextEncoders for "dev" variant
# ---------------------------------------------------------------------------

def _make_encoder(variant: str = "dev") -> Flux2TextEncoders:
    cfg = OmegaConf.create({})
    return Flux2TextEncoders(cfg, variant=variant)


# ---------------------------------------------------------------------------
# Test 1: Offline mode with no local cache raises a clear, informative error
# ---------------------------------------------------------------------------

class TestOfflineWithNoCacheRaisesClearError:
    """HF_HUB_OFFLINE=1 + no local cache → OSError with model_id and cache path."""

    def test_offline_with_no_cache_raises_clear_error(self, tmp_path, monkeypatch):
        """Monkeypatch HF_HUB_OFFLINE=1, empty HF_HOME, mock from_pretrained to raise OSError."""
        # Set offline mode and redirect cache to empty tmp dir
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        encoder = _make_encoder(variant="dev")

        # The model_id for "dev" variant is Mistral
        expected_model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        assert encoder.model_id == expected_model_id

        # Mock AutoTokenizer.from_pretrained to raise FileNotFoundError
        # (simulating a missing local snapshot)
        missing_err = FileNotFoundError("No local snapshot found")

        with patch("src.models.flux.v2.text_encoder.Flux2TextEncoders._load_mistral_encoder") as mock_load:  # noqa: F841 — context-manager handle kept for documentation
            # Instead of mocking the method entirely, call the real one with
            # patched transformers internals so the error path is exercised.
            pass

        # Use the real _load_mistral_encoder but patch AutoTokenizer
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=missing_err):
            with pytest.raises(OSError) as exc_info:
                encoder.load_pretrained()

        err_msg = str(exc_info.value)

        # Must mention the model_id (or a clear substring)
        assert "Mistral-Small-3.1-24B-Instruct-2503" in err_msg or "mistralai" in err_msg, (
            f"Error message must contain the model_id. Got: {err_msg}"
        )

        # Must mention the HF cache path that was probed
        expected_cache_substr = str(tmp_path)
        assert expected_cache_substr in err_msg, (
            f"Error message must contain the HF cache path '{expected_cache_substr}'. "
            f"Got: {err_msg}"
        )

    def test_offline_tokenizer_receives_local_files_only_true(self, tmp_path, monkeypatch):
        """When HF_HUB_OFFLINE=1, from_pretrained is called with local_files_only=True."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        encoder = _make_encoder(variant="dev")

        captured_kwargs: list[dict] = []

        def _fake_from_pretrained(model_path, **kwargs):
            captured_kwargs.append(kwargs)
            # Simulate cache miss
            raise FileNotFoundError("no local cache")

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_from_pretrained):
            with pytest.raises(OSError):
                encoder.load_pretrained()

        assert len(captured_kwargs) >= 1, "from_pretrained was never called"
        assert captured_kwargs[0].get("local_files_only") is True, (
            f"Expected local_files_only=True when HF_HUB_OFFLINE=1, "
            f"got kwargs={captured_kwargs[0]}"
        )


# ---------------------------------------------------------------------------
# Test 2: Online behaviour (no HF_HUB_OFFLINE) is unchanged
# ---------------------------------------------------------------------------

class TestOnlineBehaviourUnchanged:
    """Without HF_HUB_OFFLINE set, local_files_only is NOT forced True."""

    def test_online_behaviour_local_files_only_not_forced(self, monkeypatch):
        """Without HF_HUB_OFFLINE, local_files_only should NOT be set to True."""
        # Ensure HF_HUB_OFFLINE is unset
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

        encoder = _make_encoder(variant="dev")

        captured_kwargs: list[dict] = []

        def _fake_from_pretrained_tokenizer(model_path, **kwargs):
            captured_kwargs.append({"call": "tokenizer", **kwargs})
            # Simulate success — return a mock tokenizer
            tok = MagicMock()
            tok.pad_token = None
            tok.eos_token = "<eos>"
            return tok

        def _fake_from_pretrained_model(model_path, **kwargs):
            captured_kwargs.append({"call": "model", **kwargs})
            return MagicMock()

        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake_from_pretrained_tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_fake_from_pretrained_model),
        ):
            encoder.load_pretrained()

        # Find tokenizer call
        tok_calls = [c for c in captured_kwargs if c.get("call") == "tokenizer"]
        assert tok_calls, "AutoTokenizer.from_pretrained was never called"

        tok_kwargs = tok_calls[0]
        assert tok_kwargs.get("local_files_only") is not True, (
            "local_files_only must NOT be True when HF_HUB_OFFLINE is not set. "
            f"Got: {tok_kwargs}"
        )

    def test_online_no_hf_hub_offline_is_offline_mode_false(self, monkeypatch):
        """_is_offline_mode() returns False when HF_HUB_OFFLINE is not set."""
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        assert Flux2TextEncoders._is_offline_mode() is False

    def test_offline_mode_true_for_1(self, monkeypatch):
        """_is_offline_mode() returns True when HF_HUB_OFFLINE='1'."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert Flux2TextEncoders._is_offline_mode() is True

    def test_offline_mode_true_for_true(self, monkeypatch):
        """_is_offline_mode() returns True when HF_HUB_OFFLINE='true'."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "true")
        assert Flux2TextEncoders._is_offline_mode() is True

    def test_original_error_is_chained(self, tmp_path, monkeypatch):
        """The wrapped OSError must chain the original exception (raise X from e)."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        encoder = _make_encoder(variant="dev")
        original = FileNotFoundError("original underlying error")

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=original):
            with pytest.raises(OSError) as exc_info:
                encoder.load_pretrained()

        # Python exception chaining: __cause__ should be the original error
        assert exc_info.value.__cause__ is original, (
            "Wrapped error must chain the original exception via 'raise X from e'"
        )


# ---------------------------------------------------------------------------
# Test 3: Qwen variant also honours offline fallback
# ---------------------------------------------------------------------------

class TestQwenOfflineFallback:
    """Qwen variant also passes local_files_only=True when HF_HUB_OFFLINE=1."""

    def test_qwen_offline_local_files_only_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        encoder = _make_encoder(variant="klein-4b")
        assert "Qwen" in encoder.model_id

        captured_kwargs: list[dict] = []

        def _fake(model_path, **kwargs):
            captured_kwargs.append(kwargs)
            raise FileNotFoundError("no local qwen cache")

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=_fake):
            with pytest.raises(OSError) as exc_info:
                encoder.load_pretrained()

        err_msg = str(exc_info.value)
        assert "Qwen" in err_msg or "Qwen3-4B" in err_msg, (
            f"Error message must mention Qwen model. Got: {err_msg}"
        )
        assert len(captured_kwargs) >= 1
        assert captured_kwargs[0].get("local_files_only") is True
