#!/usr/bin/env python
"""Verify the diff_base reorganization for model-agnostic training."""
import sys
import os
os.environ['TMPDIR'] = '/n/scratch/users/f/fas994/tmp'
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf

# head_dim = 96/4 = 24. axes_dims must sum to head_dim.
SMALL_V1_CONFIG = OmegaConf.create({
    'hidden_size': 96, 'num_attention_heads': 4, 'in_channels': 32,
    'num_layers': 1, 'num_single_layers': 1,
    'pooled_projection_dim': 32, 'joint_attention_dim': 48,
    'axes_dims_rope': [4, 10, 10],  # sum=24=head_dim for v1
})

SMALL_V2_CONFIG = OmegaConf.create({
    'hidden_size': 96, 'num_attention_heads': 4, 'in_channels': 32,
    'num_layers': 1, 'num_single_layers': 1,
    'pooled_projection_dim': 32, 'joint_attention_dim': 48,
    'mlp_ratio': 2.0,
    'axes_dims_rope': [6, 6, 6, 6],  # sum=24=head_dim for v2
    'rope_theta': 2000,
})

SMALL_CONFIG = SMALL_V1_CONFIG  # backward compat

B, img_seq, txt_seq = 1, 16, 8


def test_base_import():
    print('Test 1: FluxTransformerBase import...')
    from src.models.flux.base_transformer import FluxTransformerBase
    assert FluxTransformerBase is not None
    print('  PASSED')


def test_factory():
    print('Test 2: create_flux_transformer factory...')
    from src.models.flux import create_flux_transformer, FluxTransformerBase

    t1 = create_flux_transformer('v1', SMALL_CONFIG, 'dev')
    assert isinstance(t1, FluxTransformerBase)
    print(f'  v1: {type(t1).__name__}, hidden={t1.hidden_size}, in_ch={t1.in_channels}')

    t2 = create_flux_transformer('v2', SMALL_V2_CONFIG, 'dev')
    assert isinstance(t2, FluxTransformerBase)
    print(f'  v2: {type(t2).__name__}, hidden={t2.hidden_size}, in_ch={t2.in_channels}')
    print('  PASSED')


def test_unified_conditioning():
    print('Test 3: Unified conditioning...')
    from src.models.flux.conditioning import create_position_ids, rearrange_latent_to_sequence

    ids_v1 = create_position_ids('v1', 2, 4, 4, 'cpu', torch.float32)
    assert ids_v1.shape == (2, 16, 3), f'Got {ids_v1.shape}'
    ids_v2 = create_position_ids('v2', 2, 4, 4, 'cpu', torch.float32)
    assert ids_v2.shape == (2, 16, 3), f'Got {ids_v2.shape}'

    lat = torch.randn(2, 8, 4, 4)
    seq = rearrange_latent_to_sequence(lat, patch_size=2)
    assert seq.shape == (2, 4, 32)
    print('  PASSED')


def test_v1_forward():
    print('Test 4: Flux1Transformer forward...')
    from src.models.flux import create_flux_transformer
    t = create_flux_transformer('v1', SMALL_CONFIG, 'dev')
    out = t(
        hidden_states=torch.randn(B, img_seq, 32),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(B, txt_seq, 48),
        pooled_projections=torch.randn(B, 32),
        guidance=torch.tensor([3.5]),
    )
    assert out.shape == (B, img_seq, 32), f'Got {out.shape}'
    print(f'  Output shape: {out.shape}')
    print('  PASSED')


def test_v2_forward():
    print('Test 5: Flux2Transformer forward...')
    from src.models.flux import create_flux_transformer
    t = create_flux_transformer('v2', SMALL_V2_CONFIG, 'dev')
    out = t(
        hidden_states=torch.randn(B, img_seq, 32),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(B, txt_seq, 48),
        pooled_projections=torch.randn(B, 32),
        guidance=torch.tensor([3.5]),
    )
    assert out.shape == (B, img_seq, 32), f'Got {out.shape}'
    print(f'  Output shape: {out.shape}')
    print('  PASSED')


def test_v1_repa():
    print('Test 6: return_hidden_states_at (v1)...')
    from src.models.flux import create_flux_transformer
    t = create_flux_transformer('v1', SMALL_CONFIG, 'dev')
    out, captured = t(
        hidden_states=torch.randn(B, img_seq, 32),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(B, txt_seq, 48),
        pooled_projections=torch.randn(B, 32),
        return_hidden_states_at=[0],
    )
    assert 0 in captured
    print(f'  Captured block 0: {captured[0].shape}')
    print('  PASSED')


def test_v2_repa():
    print('Test 7: return_hidden_states_at (v2)...')
    from src.models.flux import create_flux_transformer
    t = create_flux_transformer('v2', SMALL_V2_CONFIG, 'dev')
    out, captured = t(
        hidden_states=torch.randn(B, img_seq, 32),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(B, txt_seq, 48),
        pooled_projections=torch.randn(B, 32),
        guidance=torch.tensor([3.5]),
        return_hidden_states_at=[0],
    )
    assert 0 in captured
    print(f'  Captured block 0: {captured[0].shape}')
    print('  PASSED')


def test_v2_state_dict_keys():
    print('Test 8: Flux2 state dict key naming...')
    from src.models.flux import create_flux_transformer
    t = create_flux_transformer('v2', SMALL_V2_CONFIG, 'dev')
    sd = t.state_dict()

    expected_prefixes = [
        'transformer_blocks.0.attn.',
        'single_transformer_blocks.0.attn.',
        'time_guidance_embed.',
        'double_stream_modulation_img.',
        'double_stream_modulation_txt.',
        'single_stream_modulation.',
        'x_embedder.',
        'context_embedder.',
        'norm_out.',
        'proj_out.',
        'pooled_text_embed.',
    ]
    for prefix in expected_prefixes:
        matching = [k for k in sd.keys() if k.startswith(prefix)]
        assert len(matching) > 0, f'No keys with prefix {prefix}'
        print(f'  {prefix}*: {len(matching)} keys')

    # Verify HF-specific keys
    assert 'transformer_blocks.0.ff.linear_in.weight' in sd
    assert 'transformer_blocks.0.ff.linear_out.weight' in sd
    assert 'single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight' in sd
    assert 'single_transformer_blocks.0.attn.to_out.weight' in sd
    assert 'transformer_blocks.0.attn.add_q_proj.weight' in sd
    assert 'transformer_blocks.0.attn.to_out.0.weight' in sd

    # Verify NO old-style keys
    assert not any('joint_blocks' in k for k in sd.keys()), 'Found old joint_blocks keys!'
    assert not any('single_blocks' in k and 'single_transformer_blocks' not in k for k in sd.keys())
    print('  PASSED')


def test_v2_block_keys_detail():
    print('Test 9: Flux2 block key details...')
    from src.models.flux import create_flux_transformer
    t = create_flux_transformer('v2', SMALL_V2_CONFIG, 'dev')
    sd = t.state_dict()

    double_keys = sorted(k for k in sd if k.startswith('transformer_blocks.0.'))
    single_keys = sorted(k for k in sd if k.startswith('single_transformer_blocks.0.'))

    print(f'  Double block ({len(double_keys)} keys):')
    for k in double_keys:
        print(f'    {k} {list(sd[k].shape)}')

    print(f'  Single block ({len(single_keys)} keys):')
    for k in single_keys:
        print(f'    {k} {list(sd[k].shape)}')
    print('  PASSED')


def test_kontext():
    print('Test 10: Kontext conditioning...')
    from src.models.flux import create_flux_transformer
    from src.models.flux.conditioning import create_position_ids

    ref_seq = torch.randn(B, img_seq, 32)
    ref_ids = create_position_ids('v1', B, 4, 4, 'cpu', torch.float32, time_offset=1.0)

    t1 = create_flux_transformer('v1', SMALL_CONFIG, 'dev')
    out1 = t1(
        hidden_states=torch.randn(B, img_seq, 32),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(B, txt_seq, 48),
        pooled_projections=torch.randn(B, 32),
        img_cond_seq=ref_seq, img_cond_seq_ids=ref_ids,
    )
    assert out1.shape[1] == img_seq * 2
    print(f'  v1 Kontext: {out1.shape}')

    t2 = create_flux_transformer('v2', SMALL_V2_CONFIG, 'dev')
    out2 = t2(
        hidden_states=torch.randn(B, img_seq, 32),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(B, txt_seq, 48),
        pooled_projections=torch.randn(B, 32),
        guidance=torch.tensor([3.5]),
        img_cond_seq=ref_seq, img_cond_seq_ids=ref_ids,
    )
    assert out2.shape[1] == img_seq * 2
    print(f'  v2 Kontext: {out2.shape}')
    print('  PASSED')


def test_backward_compat():
    print('Test 11: Backward compatibility...')
    from src.models.flux.components.embeddings import get_timestep_embedding, MLPEmbedder
    from src.models.flux.components.embeddings import create_image_position_ids

    emb = get_timestep_embedding(torch.tensor([0.5, 1.0]))
    assert emb.shape == (2, 256)

    mlp = MLPEmbedder(256, 64)
    out = mlp(emb)
    assert out.shape == (2, 64)

    ids = create_image_position_ids(2, 4, 4, 'cpu', torch.float32)
    assert ids.shape == (2, 16, 3)

    # Old-style imports still work
    from src.models.flux import Flux1Transformer, Flux2Transformer, FluxTransformer
    assert FluxTransformer is Flux1Transformer
    print('  PASSED')


if __name__ == '__main__':
    tests = [
        test_base_import,
        test_factory,
        test_unified_conditioning,
        test_v1_forward,
        test_v2_forward,
        test_v1_repa,
        test_v2_repa,
        test_v2_state_dict_keys,
        test_v2_block_keys_detail,
        test_kontext,
        test_backward_compat,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            import traceback
            failed.append((test.__name__, str(e)))
            traceback.print_exc()

    print()
    print('=' * 60)
    if failed:
        print(f'FAILED: {len(failed)}/{len(tests)}')
        for name, err in failed:
            print(f'  {name}: {err}')
    else:
        print(f'ALL {len(tests)} TESTS PASSED!')
    print('=' * 60)
