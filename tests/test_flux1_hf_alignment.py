"""HuggingFace alignment tests for FLUX.1 transformer.

These tests verify that our Flux1Transformer produces state dict keys
compatible with HuggingFace's FluxTransformer2DModel for direct weight loading.

Key mappings verified:
- transformer_blocks (not joint_blocks)
- single_transformer_blocks (not single_blocks)
- time_text_embed.timestep_embedder/guidance_embedder/text_embedder
- Blocks use to_q/to_k/to_v (not combined to_qkv)
- attn.to_out.0 (via ModuleList, not to_out_image)
- ff.net.0.proj, ff.net.2 (structured MLP)
"""

import pytest
import torch
import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.flux.v1.transformer import Flux1Transformer


def create_test_config():
    """Create minimal config for testing."""
    return OmegaConf.create({
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_layers": 19,
        "num_single_layers": 38,
        "in_channels": 64,
        "pooled_projection_dim": 768,
        "guidance_embeds": True,
        "joint_attention_dim": 4096,  # T5 text encoder dimension
    })


class TestStateKeyAlignment:
    """Verify state dict keys match HuggingFace FluxTransformer2DModel."""

    @pytest.fixture
    def model(self):
        """Create Flux1Transformer for testing."""
        config = create_test_config()
        return Flux1Transformer(config, variant="dev")

    def test_transformer_blocks_naming(self, model):
        """Verify joint blocks use 'transformer_blocks' naming."""
        keys = set(model.state_dict().keys())

        # Should have transformer_blocks, not joint_blocks
        transformer_block_keys = [k for k in keys if "transformer_blocks." in k]
        joint_block_keys = [k for k in keys if "joint_blocks." in k]

        assert len(transformer_block_keys) > 0, "Should have transformer_blocks keys"
        assert len(joint_block_keys) == 0, "Should NOT have joint_blocks keys"

        # Check correct block indices exist
        assert any("transformer_blocks.0." in k for k in keys), "Block 0 should exist"
        assert any("transformer_blocks.18." in k for k in keys), "Block 18 should exist (19 blocks total)"

    def test_single_transformer_blocks_naming(self, model):
        """Verify single blocks use 'single_transformer_blocks' naming."""
        keys = set(model.state_dict().keys())

        # Should have single_transformer_blocks, not single_blocks
        single_transformer_block_keys = [k for k in keys if "single_transformer_blocks." in k]
        single_block_keys = [k for k in keys if k.startswith("single_blocks.")]

        assert len(single_transformer_block_keys) > 0, "Should have single_transformer_blocks keys"
        assert len(single_block_keys) == 0, "Should NOT have single_blocks keys"

        # Check correct block indices exist
        assert any("single_transformer_blocks.0." in k for k in keys), "Block 0 should exist"
        assert any("single_transformer_blocks.37." in k for k in keys), "Block 37 should exist (38 blocks total)"

    def test_joint_attention_qkv_separate(self, model):
        """Verify joint blocks use separate Q/K/V projections."""
        keys = set(model.state_dict().keys())

        # Should have separate to_q, to_k, to_v (not combined to_qkv)
        has_to_q = any("transformer_blocks.0.attn.to_q." in k for k in keys)
        has_to_k = any("transformer_blocks.0.attn.to_k." in k for k in keys)
        has_to_v = any("transformer_blocks.0.attn.to_v." in k for k in keys)

        assert has_to_q, "Should have attn.to_q"
        assert has_to_k, "Should have attn.to_k"
        assert has_to_v, "Should have attn.to_v"

        # Should NOT have combined to_qkv
        has_to_qkv_image = any("to_qkv_image" in k for k in keys)
        has_to_qkv_text = any("to_qkv_text" in k for k in keys)

        assert not has_to_qkv_image, "Should NOT have combined to_qkv_image"
        assert not has_to_qkv_text, "Should NOT have combined to_qkv_text"

    def test_joint_attention_text_projections(self, model):
        """Verify joint blocks use add_*_proj naming for text."""
        keys = set(model.state_dict().keys())

        # Should have add_q_proj, add_k_proj, add_v_proj for text
        has_add_q = any("transformer_blocks.0.attn.add_q_proj." in k for k in keys)
        has_add_k = any("transformer_blocks.0.attn.add_k_proj." in k for k in keys)
        has_add_v = any("transformer_blocks.0.attn.add_v_proj." in k for k in keys)

        assert has_add_q, "Should have attn.add_q_proj"
        assert has_add_k, "Should have attn.add_k_proj"
        assert has_add_v, "Should have attn.add_v_proj"

    def test_joint_attention_output_naming(self, model):
        """Verify joint blocks use to_out.0 and to_add_out naming."""
        keys = set(model.state_dict().keys())

        # Should have to_out.0 (ModuleList) for image output
        has_to_out_0 = any("transformer_blocks.0.attn.to_out.0." in k for k in keys)
        assert has_to_out_0, "Should have attn.to_out.0"

        # Should have to_add_out for text output
        has_to_add_out = any("transformer_blocks.0.attn.to_add_out." in k for k in keys)
        assert has_to_add_out, "Should have attn.to_add_out"

        # Should NOT have old naming
        has_to_out_image = any("to_out_image" in k for k in keys)
        has_to_out_text = any("to_out_text" in k for k in keys)

        assert not has_to_out_image, "Should NOT have to_out_image"
        assert not has_to_out_text, "Should NOT have to_out_text"

    def test_joint_block_norm_naming(self, model):
        """Verify joint blocks use norm1/norm1_context naming."""
        keys = set(model.state_dict().keys())

        # Should have norm1 (image) and norm1_context (text)
        has_norm1 = any("transformer_blocks.0.norm1.linear." in k for k in keys)
        has_norm1_context = any("transformer_blocks.0.norm1_context.linear." in k for k in keys)

        assert has_norm1, "Should have norm1"
        assert has_norm1_context, "Should have norm1_context"

        # Should NOT have old naming
        has_norm1_img = any("norm1_img" in k for k in keys)
        has_norm1_txt = any("norm1_txt" in k for k in keys)

        assert not has_norm1_img, "Should NOT have norm1_img"
        assert not has_norm1_txt, "Should NOT have norm1_txt"

    def test_joint_block_ff_naming(self, model):
        """Verify joint blocks use ff.net structure."""
        keys = set(model.state_dict().keys())

        # Should have ff.net.0.proj (GELU with projection)
        has_ff_net_0_proj = any("transformer_blocks.0.ff.net.0.proj." in k for k in keys)
        assert has_ff_net_0_proj, "Should have ff.net.0.proj"

        # Should have ff.net.2 (output linear)
        has_ff_net_2 = any("transformer_blocks.0.ff.net.2." in k for k in keys)
        assert has_ff_net_2, "Should have ff.net.2"

        # Should have ff_context.net.0.proj for text MLP
        has_ff_context = any("transformer_blocks.0.ff_context.net.0.proj." in k for k in keys)
        assert has_ff_context, "Should have ff_context.net.0.proj"

        # Should NOT have old naming
        has_mlp_img = any("mlp_img" in k for k in keys)
        has_mlp_txt = any("mlp_txt" in k for k in keys)

        assert not has_mlp_img, "Should NOT have mlp_img"
        assert not has_mlp_txt, "Should NOT have mlp_txt"

    def test_single_attention_qkv_separate(self, model):
        """Verify single blocks use separate Q/K/V projections."""
        keys = set(model.state_dict().keys())

        # Should have separate to_q, to_k, to_v in single blocks
        has_to_q = any("single_transformer_blocks.0.attn.to_q." in k for k in keys)
        has_to_k = any("single_transformer_blocks.0.attn.to_k." in k for k in keys)
        has_to_v = any("single_transformer_blocks.0.attn.to_v." in k for k in keys)

        assert has_to_q, "Should have single attn.to_q"
        assert has_to_k, "Should have single attn.to_k"
        assert has_to_v, "Should have single attn.to_v"

        # Should have to_out.0 in single blocks
        has_to_out_0 = any("single_transformer_blocks.0.attn.to_out.0." in k for k in keys)
        assert has_to_out_0, "Should have single attn.to_out.0"

    def test_single_block_mlp_naming(self, model):
        """Verify single blocks use proj_mlp and proj_out naming."""
        keys = set(model.state_dict().keys())

        # Should have proj_mlp and proj_out
        has_proj_mlp = any("single_transformer_blocks.0.proj_mlp." in k for k in keys)
        has_proj_out = any("single_transformer_blocks.0.proj_out." in k for k in keys)

        assert has_proj_mlp, "Should have proj_mlp"
        assert has_proj_out, "Should have proj_out"

    def test_time_text_embed_structure(self, model):
        """Verify time_text_embed has correct HF structure."""
        keys = set(model.state_dict().keys())

        # Should have time_text_embed.timestep_embedder
        has_timestep = any("time_text_embed.timestep_embedder.linear_1." in k for k in keys)
        has_timestep_2 = any("time_text_embed.timestep_embedder.linear_2." in k for k in keys)
        assert has_timestep, "Should have time_text_embed.timestep_embedder.linear_1"
        assert has_timestep_2, "Should have time_text_embed.timestep_embedder.linear_2"

        # Should have time_text_embed.text_embedder
        has_text = any("time_text_embed.text_embedder.linear_1." in k for k in keys)
        has_text_2 = any("time_text_embed.text_embedder.linear_2." in k for k in keys)
        assert has_text, "Should have time_text_embed.text_embedder.linear_1"
        assert has_text_2, "Should have time_text_embed.text_embedder.linear_2"

        # Should have time_text_embed.guidance_embedder (for dev variant)
        has_guidance = any("time_text_embed.guidance_embedder.linear_1." in k for k in keys)
        assert has_guidance, "Should have time_text_embed.guidance_embedder.linear_1"

        # Should NOT have old naming
        has_time_embed = any(k.startswith("time_embed.") for k in keys)
        has_guidance_embed = any(k.startswith("guidance_embed.") for k in keys)
        has_pooled_text = any(k.startswith("pooled_text_embed.") for k in keys)

        assert not has_time_embed, "Should NOT have standalone time_embed"
        assert not has_guidance_embed, "Should NOT have standalone guidance_embed"
        assert not has_pooled_text, "Should NOT have standalone pooled_text_embed"

    def test_context_embedder_exists(self, model):
        """Verify context_embedder exists for text projection."""
        keys = set(model.state_dict().keys())

        # Should have context_embedder (projects T5 4096 -> hidden 3072)
        has_context_weight = any("context_embedder.weight" in k for k in keys)
        has_context_bias = any("context_embedder.bias" in k for k in keys)

        assert has_context_weight, "Should have context_embedder.weight"
        assert has_context_bias, "Should have context_embedder.bias"

    def test_schnell_no_guidance_embedder(self):
        """Verify schnell variant has no guidance embedder."""
        config = OmegaConf.create({
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "in_channels": 64,
            "pooled_projection_dim": 768,
            "guidance_embeds": False,
        })
        model = Flux1Transformer(config, variant="schnell")
        keys = set(model.state_dict().keys())

        # Should NOT have guidance_embedder
        has_guidance = any("guidance_embedder" in k for k in keys)
        assert not has_guidance, "schnell should NOT have guidance_embedder"


class TestPEFTTargetModules:
    """Verify PEFT can find LoRA target modules."""

    @pytest.fixture
    def model(self):
        """Create Flux1Transformer for testing."""
        config = create_test_config()
        return Flux1Transformer(config, variant="dev")

    def test_to_q_modules_exist(self, model):
        """Verify to_q modules exist and can be found."""
        to_q_modules = []
        for name, module in model.named_modules():
            if name.endswith(".to_q"):
                to_q_modules.append(name)

        assert len(to_q_modules) > 0, "Should have to_q modules"
        # 19 joint blocks + 38 single blocks = 57 to_q modules
        assert len(to_q_modules) == 57, f"Expected 57 to_q modules, got {len(to_q_modules)}"

    def test_to_k_modules_exist(self, model):
        """Verify to_k modules exist and can be found."""
        to_k_modules = []
        for name, module in model.named_modules():
            if name.endswith(".to_k"):
                to_k_modules.append(name)

        assert len(to_k_modules) > 0, "Should have to_k modules"
        assert len(to_k_modules) == 57, f"Expected 57 to_k modules, got {len(to_k_modules)}"

    def test_to_v_modules_exist(self, model):
        """Verify to_v modules exist and can be found."""
        to_v_modules = []
        for name, module in model.named_modules():
            if name.endswith(".to_v"):
                to_v_modules.append(name)

        assert len(to_v_modules) > 0, "Should have to_v modules"
        assert len(to_v_modules) == 57, f"Expected 57 to_v modules, got {len(to_v_modules)}"

    def test_to_out_0_modules_exist(self, model):
        """Verify to_out.0 modules exist and can be found."""
        to_out_modules = []
        for name, module in model.named_modules():
            if name.endswith(".to_out.0"):
                to_out_modules.append(name)

        assert len(to_out_modules) > 0, "Should have to_out.0 modules"
        # 19 joint blocks + 38 single blocks = 57 to_out.0 modules
        assert len(to_out_modules) == 57, f"Expected 57 to_out.0 modules, got {len(to_out_modules)}"

    def test_peft_target_modules_findable(self, model):
        """Verify all standard PEFT targets are findable."""
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        for target in target_modules:
            found = False
            for name, module in model.named_modules():
                if name.endswith(target):
                    found = True
                    break
            assert found, f"Target module '{target}' not found"

    def test_add_proj_modules_exist(self, model):
        """Verify add_*_proj modules exist for text in joint blocks."""
        add_q_modules = []
        add_k_modules = []
        add_v_modules = []

        for name, module in model.named_modules():
            if name.endswith(".add_q_proj"):
                add_q_modules.append(name)
            if name.endswith(".add_k_proj"):
                add_k_modules.append(name)
            if name.endswith(".add_v_proj"):
                add_v_modules.append(name)

        # 19 joint blocks have add_*_proj modules
        assert len(add_q_modules) == 19, f"Expected 19 add_q_proj, got {len(add_q_modules)}"
        assert len(add_k_modules) == 19, f"Expected 19 add_k_proj, got {len(add_k_modules)}"
        assert len(add_v_modules) == 19, f"Expected 19 add_v_proj, got {len(add_v_modules)}"


class TestStateDictShape:
    """Verify state dict tensor shapes are correct."""

    @pytest.fixture
    def model(self):
        """Create Flux1Transformer for testing."""
        config = create_test_config()
        return Flux1Transformer(config, variant="dev")

    def test_qkv_projection_shapes(self, model):
        """Verify Q/K/V projections have correct shapes."""
        state_dict = model.state_dict()
        hidden_size = 3072

        # Check joint block to_q shape
        q_weight = state_dict["transformer_blocks.0.attn.to_q.weight"]
        assert q_weight.shape == (hidden_size, hidden_size), f"to_q wrong shape: {q_weight.shape}"

        # Check joint block to_k shape
        k_weight = state_dict["transformer_blocks.0.attn.to_k.weight"]
        assert k_weight.shape == (hidden_size, hidden_size), f"to_k wrong shape: {k_weight.shape}"

        # Check joint block to_v shape
        v_weight = state_dict["transformer_blocks.0.attn.to_v.weight"]
        assert v_weight.shape == (hidden_size, hidden_size), f"to_v wrong shape: {v_weight.shape}"

    def test_context_embedder_shape(self, model):
        """Verify context_embedder projects from T5 dim (4096) to hidden (3072)."""
        state_dict = model.state_dict()
        hidden_size = 3072
        joint_attention_dim = 4096  # T5 dimension

        # context_embedder: [hidden_size, joint_attention_dim]
        ctx_weight = state_dict["context_embedder.weight"]
        ctx_bias = state_dict["context_embedder.bias"]
        assert ctx_weight.shape == (hidden_size, joint_attention_dim), f"context_embedder.weight wrong shape: {ctx_weight.shape}"
        assert ctx_bias.shape == (hidden_size,), f"context_embedder.bias wrong shape: {ctx_bias.shape}"

    def test_output_projection_shapes(self, model):
        """Verify output projections have correct shapes."""
        state_dict = model.state_dict()
        hidden_size = 3072

        # Check to_out.0 shape
        out_weight = state_dict["transformer_blocks.0.attn.to_out.0.weight"]
        assert out_weight.shape == (hidden_size, hidden_size), f"to_out.0 wrong shape: {out_weight.shape}"

        # Check to_add_out shape
        add_out_weight = state_dict["transformer_blocks.0.attn.to_add_out.weight"]
        assert add_out_weight.shape == (hidden_size, hidden_size), f"to_add_out wrong shape: {add_out_weight.shape}"

    def test_ff_projection_shapes(self, model):
        """Verify feed-forward projections have correct shapes."""
        state_dict = model.state_dict()
        hidden_size = 3072
        mlp_hidden = int(hidden_size * 4.0)  # mlp_ratio = 4.0

        # Check ff.net.0.proj (input projection)
        ff_in = state_dict["transformer_blocks.0.ff.net.0.proj.weight"]
        assert ff_in.shape == (mlp_hidden, hidden_size), f"ff.net.0.proj wrong shape: {ff_in.shape}"

        # Check ff.net.2 (output projection)
        ff_out = state_dict["transformer_blocks.0.ff.net.2.weight"]
        assert ff_out.shape == (hidden_size, mlp_hidden), f"ff.net.2 wrong shape: {ff_out.shape}"

    def test_single_block_proj_out_shape(self, model):
        """Verify single block proj_out takes concatenated attn+mlp input."""
        state_dict = model.state_dict()
        hidden_size = 3072
        mlp_hidden = int(hidden_size * 4.0)  # mlp_ratio = 4.0

        # proj_out input is concatenated [attn_out, mlp_out] = [3072, 12288] = 15360
        proj_out = state_dict["single_transformer_blocks.0.proj_out.weight"]
        expected_in = hidden_size + mlp_hidden  # 3072 + 12288 = 15360
        assert proj_out.shape == (hidden_size, expected_in), f"proj_out wrong shape: {proj_out.shape}, expected ({hidden_size}, {expected_in})"


class TestForwardPass:
    """Verify forward pass works with aligned naming."""

    @pytest.fixture
    def model(self):
        """Create Flux1Transformer for testing."""
        config = create_test_config()
        return Flux1Transformer(config, variant="dev")

    def test_forward_basic(self, model):
        """Test basic forward pass works."""
        batch_size = 1
        seq_len = 64  # 8x8 patches
        in_channels = 64
        txt_seq_len = 77
        joint_attention_dim = 4096  # T5 dimension
        pooled_dim = 768

        hidden_states = torch.randn(batch_size, seq_len, in_channels)
        timestep = torch.tensor([0.5])
        encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim)
        pooled_projections = torch.randn(batch_size, pooled_dim)
        guidance = torch.tensor([3.5])

        # Should not raise
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            guidance=guidance,
        )

        # Output should have same spatial dimensions as input
        assert output.shape == (batch_size, seq_len, in_channels)

    def test_forward_schnell_no_guidance(self):
        """Test forward pass for schnell (no guidance)."""
        config = OmegaConf.create({
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "in_channels": 64,
            "pooled_projection_dim": 768,
            "guidance_embeds": False,
            "joint_attention_dim": 4096,
        })
        model = Flux1Transformer(config, variant="schnell")

        batch_size = 1
        seq_len = 64
        in_channels = 64
        txt_seq_len = 77
        joint_attention_dim = 4096  # T5 dimension
        pooled_dim = 768

        hidden_states = torch.randn(batch_size, seq_len, in_channels)
        timestep = torch.tensor([0.5])
        encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim)
        pooled_projections = torch.randn(batch_size, pooled_dim)

        # Should not raise even without guidance
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            guidance=None,
        )

        assert output.shape == (batch_size, seq_len, in_channels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
