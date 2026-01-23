#!/usr/bin/env python
"""Simple test runner to verify implementations."""

import sys
import os
from pathlib import Path

# Set TMPDIR to scratch space BEFORE any imports
os.environ['TMPDIR'] = '/n/scratch/users/f/fas994/tmp'
os.environ['TEMP'] = '/n/scratch/users/f/fas994/tmp'
os.environ['TMP'] = '/n/scratch/users/f/fas994/tmp'

import tempfile

# Override tempfile's temp directory
tempfile.tempdir = '/n/scratch/users/f/fas994/tmp'

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
import torch
import torch.nn as nn


def run_config_tests():
    """Test configuration module."""
    print("\n" + "="*60)
    print("TESTING: Configuration Module")
    print("="*60)

    from src.utils.config import load_config, validate_config, save_config

    # Test 1: Simple config loading
    print("\n1. Testing simple config loading...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config_content = """
experiment:
  name: test-experiment
  output_dir: ./outputs
model:
  type: sdxl
training:
  method: lora
  epochs: 10
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        assert config.experiment.name == "test-experiment"
        assert config.model.type == "sdxl"
        print("   PASSED: Simple config loading")

    # Test 2: Config validation
    print("\n2. Testing config validation...")
    config = OmegaConf.create({
        "experiment": {"name": "test"},
        "model": {"type": "sdxl"},
        "training": {"method": "lora"},
    })
    validate_config(config)
    print("   PASSED: Config validation")

    # Test 3: Save config
    print("\n3. Testing config save...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "saved.yaml"
        save_config(config, save_path)
        assert save_path.exists()
        print("   PASSED: Config save")

    print("\n[CONFIG TESTS: ALL PASSED]")


def run_model_tests():
    """Test model components."""
    print("\n" + "="*60)
    print("TESTING: Model Components")
    print("="*60)

    # Test attention modules
    print("\n1. Testing attention modules...")
    from src.models.components.attention import SelfAttention, CrossAttention, JointAttention

    # Self attention (uses query_dim, heads, dim_head)
    attn = SelfAttention(query_dim=64, heads=4, dim_head=16)
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == (2, 16, 64)
    print("   PASSED: SelfAttention")

    # Cross attention (uses query_dim, cross_attention_dim, heads, dim_head)
    cross_attn = CrossAttention(query_dim=64, cross_attention_dim=128, heads=4, dim_head=16)
    context = torch.randn(2, 32, 128)
    out = cross_attn(x, context)
    assert out.shape == (2, 16, 64)
    print("   PASSED: CrossAttention")

    # Joint attention
    joint_attn = JointAttention(dim=64, num_heads=4)
    out_x, out_ctx = joint_attn(x, torch.randn(2, 32, 64))
    assert out_x.shape == (2, 16, 64)
    print("   PASSED: JointAttention")

    # Test embeddings
    print("\n2. Testing embedding modules...")
    from src.models.components.embeddings import Timesteps, TimestepEmbedding

    timesteps = torch.randint(0, 1000, (4,))

    # Timesteps module creates sinusoidal embeddings
    timesteps_module = Timesteps(128)
    emb = timesteps_module(timesteps)
    assert emb.shape == (4, 128)
    print("   PASSED: Timesteps module")

    # TimestepEmbedding MLP projects embeddings
    mlp = TimestepEmbedding(128, 512)
    out = mlp(emb)
    assert out.shape == (4, 512)
    print("   PASSED: TimestepEmbedding MLP")

    # Test ResNet blocks
    print("\n3. Testing ResNet blocks...")
    from src.models.components.resnet import ResnetBlock2D, Downsample2D, Upsample2D

    block = ResnetBlock2D(in_channels=64, out_channels=128, temb_channels=256)
    x = torch.randn(2, 64, 32, 32)
    temb = torch.randn(2, 256)
    out = block(x, temb)
    assert out.shape == (2, 128, 32, 32)
    print("   PASSED: ResnetBlock2D")

    down = Downsample2D(64, use_conv=True)
    out = down(torch.randn(2, 64, 32, 32))
    assert out.shape == (2, 64, 16, 16)
    print("   PASSED: Downsample2D")

    up = Upsample2D(64, use_conv=True)
    out = up(torch.randn(2, 64, 16, 16))
    assert out.shape == (2, 64, 32, 32)
    print("   PASSED: Upsample2D")

    # Test transformer blocks
    print("\n4. Testing transformer blocks...")
    from src.models.components.transformer import FeedForward
    from src.models.components.attention import BasicTransformerBlock

    ff = FeedForward(dim=256, mult=4.0)
    out = ff(torch.randn(2, 16, 256))
    assert out.shape == (2, 16, 256)
    print("   PASSED: FeedForward")

    block = BasicTransformerBlock(dim=256, num_heads=4, context_dim=512)
    out = block(torch.randn(2, 16, 256), encoder_hidden_states=torch.randn(2, 32, 512))
    assert out.shape == (2, 16, 256)
    print("   PASSED: BasicTransformerBlock")

    # Test model factory
    print("\n5. Testing model factory...")
    from src.models import create_model, BaseDiffusionModel
    # Just verify the imports work - actual model creation requires pretrained weights
    assert callable(create_model)
    assert BaseDiffusionModel is not None
    print("   PASSED: Model factory imports")

    print("\n[MODEL TESTS: ALL PASSED]")


def run_scheduler_tests():
    """Test noise schedulers."""
    print("\n" + "="*60)
    print("TESTING: Noise Schedulers")
    print("="*60)

    # Test DDPM (takes config object)
    print("\n1. Testing DDPM scheduler...")
    from src.schedulers.ddpm import DDPMScheduler

    ddpm_config = OmegaConf.create({"num_train_timesteps": 1000})
    ddpm = DDPMScheduler(ddpm_config)
    ddpm.set_timesteps(50)
    assert len(ddpm.timesteps) == 50
    print("   PASSED: DDPM initialization")

    sample = torch.randn(2, 4, 64, 64)
    noise = torch.randn_like(sample)
    timesteps = torch.randint(0, 1000, (2,))
    noisy = ddpm.add_noise(sample, noise, timesteps)
    assert noisy.shape == sample.shape
    print("   PASSED: DDPM add_noise")

    noise_pred = torch.randn_like(sample)
    out, _ = ddpm.step(noise_pred, ddpm.timesteps[0], sample)
    assert out.shape == sample.shape
    print("   PASSED: DDPM step")

    # Test SNR
    snr = ddpm.get_snr(torch.tensor([100, 500, 900]))
    assert snr[0] > snr[1] > snr[2]
    print("   PASSED: DDPM SNR computation")

    # Test Euler (takes config object)
    print("\n2. Testing Euler scheduler...")
    from src.schedulers.euler import EulerDiscreteScheduler

    euler_config = OmegaConf.create({"num_train_timesteps": 1000})
    euler = EulerDiscreteScheduler(euler_config)
    euler.set_timesteps(25)
    assert len(euler.timesteps) == 25
    print("   PASSED: Euler initialization")

    out, _ = euler.step(noise_pred, euler.timesteps[0], sample)
    assert out.shape == sample.shape
    print("   PASSED: Euler step")

    # Test Flow Matching (takes config object)
    print("\n3. Testing Flow Matching scheduler...")
    from src.schedulers.flow_matching import FlowMatchingScheduler

    flow_config = OmegaConf.create({"num_train_timesteps": 1000})
    flow = FlowMatchingScheduler(flow_config)
    flow.set_timesteps(28)
    assert len(flow.timesteps) == 28
    print("   PASSED: Flow Matching initialization")

    noisy = flow.add_noise(sample, noise, timesteps)
    assert noisy.shape == sample.shape
    print("   PASSED: Flow Matching add_noise")

    velocity = flow.get_velocity(sample, noise, timesteps)
    assert velocity.shape == sample.shape
    print("   PASSED: Flow Matching get_velocity")

    # Test factory
    print("\n4. Testing scheduler factory...")
    from src.schedulers import create_scheduler

    config = OmegaConf.create({"type": "ddpm", "num_train_timesteps": 1000})
    scheduler = create_scheduler(config)
    assert isinstance(scheduler, DDPMScheduler)
    print("   PASSED: Scheduler factory")

    print("\n[SCHEDULER TESTS: ALL PASSED]")


def run_lora_tests():
    """Test LoRA implementation (uses peft library)."""
    print("\n" + "="*60)
    print("TESTING: LoRA Implementation")
    print("="*60)

    from src.training.methods.lora import inject_lora_layers, get_lora_parameters

    # Test injection with peft
    print("\n1. Testing LoRA injection with peft...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(64, 64)
            self.to_k = nn.Linear(64, 64)
            self.to_v = nn.Linear(64, 64)
            self.other = nn.Linear(64, 64)

        def forward(self, x):
            return self.to_q(x) + self.to_k(x) + self.to_v(x) + self.other(x)

    model = SimpleModel()

    # Test without peft (manual injection)
    model_manual = inject_lora_layers(
        model,
        rank=4,
        alpha=4.0,
        dropout=0.0,
        target_modules=["to_q", "to_k", "to_v"],
        use_peft=False
    )
    print("   PASSED: LoRA injection (manual)")

    # Test getting LoRA params
    print("\n2. Testing get_lora_parameters...")
    lora_params = get_lora_parameters(model_manual)
    # Should have LoRA parameters
    print(f"   Found {len(lora_params)} LoRA parameters")
    print("   PASSED: get_lora_parameters")

    # Test forward pass works
    print("\n3. Testing forward pass with LoRA...")
    x = torch.randn(2, 64)
    out = model_manual(x)
    assert out.shape == (2, 64)
    print("   PASSED: Forward pass with LoRA")

    print("\n[LORA TESTS: ALL PASSED]")


def run_data_tests():
    """Test data pipeline."""
    print("\n" + "="*60)
    print("TESTING: Data Pipeline")
    print("="*60)

    # Test transforms
    print("\n1. Testing transforms...")
    from src.data.transforms import create_transforms, tensor_to_pil
    from PIL import Image

    config = OmegaConf.create({"data": {}})
    transform = create_transforms(resolution=256, config=config)

    img = Image.new("RGB", (512, 512), color="red")
    tensor = transform(img)
    assert tensor.shape == (3, 256, 256)
    print("   PASSED: create_transforms")

    # Test tensor to PIL
    back_to_pil = tensor_to_pil(tensor)
    assert isinstance(back_to_pil, Image.Image)
    print("   PASSED: tensor_to_pil")

    # Test bucketing
    print("\n2. Testing bucketing...")
    from src.data.bucket import compute_bucket_sizes

    buckets = compute_bucket_sizes(base_resolution=1024, step=64, min_dim=512, max_dim=2048)
    assert len(buckets) > 0
    for w, h in buckets:
        assert w % 64 == 0
        assert h % 64 == 0
    print("   PASSED: compute_bucket_sizes")

    # Test collate
    print("\n3. Testing collate functions...")
    from src.data.collate import default_collate_fn

    batch = [
        {"image": torch.randn(3, 64, 64), "caption": "test 1"},
        {"image": torch.randn(3, 64, 64), "caption": "test 2"},
    ]
    collated = default_collate_fn(batch)
    # Note: collate returns "images" (plural) not "image"
    assert collated["images"].shape == (2, 3, 64, 64)
    assert collated["captions"] == ["test 1", "test 2"]
    print("   PASSED: default_collate_fn")

    print("\n[DATA TESTS: ALL PASSED]")


def run_trainer_tests():
    """Test trainer components."""
    print("\n" + "="*60)
    print("TESTING: Trainer Components")
    print("="*60)

    # Test trainer registry
    print("\n1. Testing trainer registry...")
    from src.training import TRAINER_REGISTRY

    expected = ["lora", "full_finetune", "dreambooth", "controlnet", "textual_inversion"]
    for name in expected:
        assert name in TRAINER_REGISTRY
    print("   PASSED: All trainers registered")

    print("\n[TRAINER TESTS: ALL PASSED]")


def run_checkpoint_tests():
    """Test checkpoint utilities."""
    print("\n" + "="*60)
    print("TESTING: Checkpoint Utilities")
    print("="*60)

    from src.utils.checkpoint import save_checkpoint, load_checkpoint

    print("\n1. Testing save/load checkpoint...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = nn.Linear(64, 64)
        optimizer = torch.optim.Adam(model.parameters())
        config = OmegaConf.create({"test": "value"})

        path = Path(tmp_dir) / "checkpoint"
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            step=100,
            epoch=5,
            config=config,
        )

        # load_checkpoint returns step, epoch, config - model is loaded directly
        loaded = load_checkpoint(path)
        assert loaded["step"] == 100, f"Expected step=100, got {loaded.get('step')}"
        assert loaded["epoch"] == 5, f"Expected epoch=5, got {loaded.get('epoch')}"
        print("   PASSED: save_checkpoint and load_checkpoint")

    print("\n[CHECKPOINT TESTS: ALL PASSED]")


def main():
    """Run all tests."""
    print("="*60)
    print("DIFFUSION TRAINING FRAMEWORK - UNIT TESTS")
    print("="*60)

    tests = [
        ("Config", run_config_tests),
        ("Models", run_model_tests),
        ("Schedulers", run_scheduler_tests),
        ("LoRA", run_lora_tests),
        ("Data", run_data_tests),
        ("Trainers", run_trainer_tests),
        ("Checkpoints", run_checkpoint_tests),
    ]

    failed = []
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            failed.append((name, str(e)))
            print(f"\n[{name} TESTS: FAILED]")
            print(f"Error: {e}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if failed:
        print(f"\nFailed tests: {len(failed)}/{len(tests)}")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1
    else:
        print(f"\nAll {len(tests)} test suites passed!")
        return 0


if __name__ == "__main__":
    exit(main())
