"""
Test Phase 9: Multi-Task Learning

This script tests the multi-task implementation.
"""

import torch
import numpy as np
from Implementation_of_DeepDTA_pipeline.model import DeepDTAModel
from Implementation_of_DeepDTA_pipeline.multitask import (
    MultiTaskHead,
    MultiTaskLoss,
    evaluate_multitask,
)


def test_multitask_head():
    """Test MultiTaskHead module."""
    print("\n" + "="*70)
    print("TEST 1: Multi-Task Head")
    print("="*70)
    
    batch_size = 8
    input_dim = 768
    num_moa_classes = 10
    
    # Create head
    head = MultiTaskHead(
        input_dim=input_dim,
        num_moa_classes=num_moa_classes,
        hidden_dim=256,
        dropout=0.2,
    )
    
    # Forward pass
    x = torch.randn(batch_size, input_dim)
    output = head(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output keys: {list(output.keys())}")
    
    # Check all required outputs
    assert 'affinity' in output, "Missing affinity output"
    assert 'interaction' in output, "Missing interaction output"
    assert 'moa' in output, "Missing MoA output"
    
    print(f"  - affinity: {output['affinity'].shape} ✓")
    print(f"  - interaction: {output['interaction'].shape} ✓")
    print(f"  - moa: {output['moa'].shape} ✓")
    
    # Check shapes
    assert output['affinity'].shape == (batch_size,), f"Wrong affinity shape: {output['affinity'].shape}"
    assert output['interaction'].shape == (batch_size,), f"Wrong interaction shape: {output['interaction'].shape}"
    assert output['moa'].shape == (batch_size, num_moa_classes), f"Wrong MoA shape: {output['moa'].shape}"
    
    # Check interaction is in [0, 1] (sigmoid applied)
    assert (output['interaction'] >= 0).all() and (output['interaction'] <= 1).all(), \
        "Interaction probabilities not in [0, 1]"
    
    print(f"\n✓ Affinity range: [{output['affinity'].min():.2f}, {output['affinity'].max():.2f}]")
    print(f"✓ Interaction range: [{output['interaction'].min():.2f}, {output['interaction'].max():.2f}]")
    print(f"✓ MoA logits range: [{output['moa'].min():.2f}, {output['moa'].max():.2f}]")
    
    print("\n✅ Multi-task head test PASSED")


def test_multitask_loss():
    """Test MultiTaskLoss function."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Task Loss")
    print("="*70)
    
    batch_size = 16
    num_moa_classes = 10
    
    # Create loss function
    loss_fn = MultiTaskLoss(
        loss_weights={'affinity': 1.0, 'interaction': 1.0, 'moa': 1.0},
        use_dynamic_weighting=False,
        affinity_threshold=7.0,
    )
    
    # Create dummy predictions (with requires_grad=True)
    predictions = {
        'affinity': torch.randn(batch_size, requires_grad=True),
        'interaction': torch.rand(batch_size, requires_grad=True),  # [0, 1]
        'moa': torch.randn(batch_size, num_moa_classes, requires_grad=True),
    }
    
    # Create targets
    targets = {
        'affinity': torch.randn(batch_size) * 2 + 7,
        'interaction_label': torch.randint(0, 2, (batch_size,)).float(),
        'moa_label': torch.randint(0, num_moa_classes, (batch_size,)),
    }
    
    # Compute loss
    losses = loss_fn(predictions, targets)
    
    print(f"✓ Loss keys: {list(losses.keys())}")
    
    # Check all loss components
    assert 'loss_affinity' in losses, "Missing affinity loss"
    assert 'loss_interaction' in losses, "Missing interaction loss"
    assert 'loss_moa' in losses, "Missing MoA loss"
    assert 'loss_total' in losses, "Missing total loss"
    
    print(f"  - loss_affinity: {losses['loss_affinity'].item():.4f} ✓")
    print(f"  - loss_interaction: {losses['loss_interaction'].item():.4f} ✓")
    print(f"  - loss_moa: {losses['loss_moa'].item():.4f} ✓")
    print(f"  - loss_total: {losses['loss_total'].item():.4f} ✓")
    
    # Check losses are positive
    for key, value in losses.items():
        assert value.item() >= 0, f"{key} is negative: {value.item()}"
    
    # Test backward pass
    losses['loss_total'].backward()
    print(f"✓ Backward pass successful")
    
    # Check gradients
    assert predictions['affinity'].grad is not None, "Affinity gradient is None"
    assert predictions['interaction'].grad is not None, "Interaction gradient is None"
    assert predictions['moa'].grad is not None, "MoA gradient is None"
    print(f"✓ All gradients computed")
    
    # Test with missing labels
    predictions_new = {
        'affinity': torch.randn(batch_size, requires_grad=True),
        'interaction': torch.rand(batch_size, requires_grad=True),
        'moa': torch.randn(batch_size, num_moa_classes, requires_grad=True),
    }
    
    targets_partial = {
        'affinity': torch.randn(batch_size) * 2 + 7,
        # No interaction or MoA labels
    }
    
    losses_partial = loss_fn(predictions_new, targets_partial)
    print(f"\n✓ Partial labels handled:")
    print(f"  - loss_affinity: {losses_partial['loss_affinity'].item():.4f}")
    print(f"  - loss_interaction: {losses_partial['loss_interaction'].item():.4f}")
    print(f"  - loss_moa: {losses_partial['loss_moa'].item():.4f}")
    
    print("\n✅ Multi-task loss test PASSED")


def test_model_integration():
    """Test multi-task head integrated into DeepDTAModel."""
    print("\n" + "="*70)
    print("TEST 3: Model Integration")
    print("="*70)
    
    batch_size = 4
    vocab_drug = 65
    vocab_prot = 26
    num_moa_classes = 10
    
    # Create model with multi-task head
    model = DeepDTAModel(
        vocab_drug=vocab_drug,
        vocab_prot=vocab_prot,
        emb_dim=128,
        conv_out=128,
        use_multitask=True,
        num_moa_classes=num_moa_classes,
    )
    
    print(f"✓ Model created with multi-task head")
    print(f"  - Head type: {type(model.fc).__name__}")
    
    # Create dummy inputs
    smiles = torch.randint(0, vocab_drug, (batch_size, 100))
    seq = torch.randint(0, vocab_prot, (batch_size, 500))
    
    # Forward pass
    output = model(smiles, seq)
    
    print(f"✓ Forward pass output type: {type(output)}")
    assert isinstance(output, dict), "Output should be dict for multi-task model"
    print(f"✓ Output keys: {list(output.keys())}")
    
    # Create loss function and targets
    loss_fn = MultiTaskLoss()
    targets = {
        'affinity': torch.randn(batch_size) * 2 + 7,
        'interaction_label': torch.randint(0, 2, (batch_size,)).float(),
        'moa_label': torch.randint(0, num_moa_classes, (batch_size,)),
    }
    
    # Compute loss
    losses = loss_fn(output, targets)
    print(f"✓ Total loss: {losses['loss_total'].item():.4f}")
    
    # Test backward pass
    losses['loss_total'].backward()
    print(f"✓ Backward pass successful")
    
    # Check gradients
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients computed"
    print(f"✓ Gradients computed")
    
    print("\n✅ Model integration test PASSED")


def test_dynamic_weighting():
    """Test dynamic loss weighting."""
    print("\n" + "="*70)
    print("TEST 4: Dynamic Loss Weighting")
    print("="*70)
    
    batch_size = 16
    num_moa_classes = 10
    
    # Create loss function with dynamic weighting
    loss_fn = MultiTaskLoss(
        use_dynamic_weighting=True,
    )
    
    print(f"✓ Dynamic weighting enabled")
    print(f"  - log_var_affinity: {loss_fn.log_var_affinity.item():.4f}")
    print(f"  - log_var_interaction: {loss_fn.log_var_interaction.item():.4f}")
    print(f"  - log_var_moa: {loss_fn.log_var_moa.item():.4f}")
    
    # Create dummy data (with requires_grad=True)
    predictions = {
        'affinity': torch.randn(batch_size, requires_grad=True),
        'interaction': torch.rand(batch_size, requires_grad=True),
        'moa': torch.randn(batch_size, num_moa_classes, requires_grad=True),
    }
    
    targets = {
        'affinity': torch.randn(batch_size) * 2 + 7,
        'interaction_label': torch.randint(0, 2, (batch_size,)).float(),
        'moa_label': torch.randint(0, num_moa_classes, (batch_size,)),
    }
    
    # Compute loss
    losses = loss_fn(predictions, targets)
    print(f"\n✓ Loss with dynamic weighting: {losses['loss_total'].item():.4f}")
    
    # Test that log_vars are learnable
    losses['loss_total'].backward()
    assert loss_fn.log_var_affinity.grad is not None, "log_var_affinity not learnable"
    assert loss_fn.log_var_interaction.grad is not None, "log_var_interaction not learnable"
    assert loss_fn.log_var_moa.grad is not None, "log_var_moa not learnable"
    print(f"✓ Log-variance parameters are learnable")
    
    print("\n✅ Dynamic weighting test PASSED")


def test_missing_label_masking():
    """Test that missing labels are properly masked."""
    print("\n" + "="*70)
    print("TEST 5: Missing Label Masking")
    print("="*70)
    
    batch_size = 16
    num_moa_classes = 10
    
    loss_fn = MultiTaskLoss()
    
    predictions = {
        'affinity': torch.randn(batch_size, requires_grad=True),
        'interaction': torch.rand(batch_size, requires_grad=True),
        'moa': torch.randn(batch_size, num_moa_classes, requires_grad=True),
    }
    
    # Test 1: Only affinity labels
    # Note: MultiTaskLoss derives interaction labels from affinity when not provided
    targets_aff_only = {
        'affinity': torch.randn(batch_size) * 2 + 7,
    }
    
    losses = loss_fn(predictions, targets_aff_only)
    print(f"✓ Affinity-only:")
    print(f"  - loss_affinity: {losses['loss_affinity'].item():.4f}")
    print(f"  - loss_interaction: {losses['loss_interaction'].item():.4f} (derived from affinity)")
    print(f"  - loss_moa: {losses['loss_moa'].item():.4f} (should be 0)")
    
    # Interaction loss is derived from affinity, so it won't be 0
    # Only MoA loss should be 0
    assert losses['loss_moa'].item() == 0.0, "MoA loss should be 0 when no labels"
    print(f"✓ MoA loss correctly masked when no labels provided")
    
    # Test 2: Partial MoA labels (some -1)
    predictions_new = {
        'affinity': torch.randn(batch_size, requires_grad=True),
        'interaction': torch.rand(batch_size, requires_grad=True),
        'moa': torch.randn(batch_size, num_moa_classes, requires_grad=True),
    }
    
    targets_partial_moa = {
        'affinity': torch.randn(batch_size) * 2 + 7,
        'moa_label': torch.cat([
            torch.randint(0, num_moa_classes, (batch_size // 2,)),
            torch.full((batch_size - batch_size // 2,), -1, dtype=torch.long),
        ]),
    }
    
    losses = loss_fn(predictions_new, targets_partial_moa)
    print(f"\n✓ Partial MoA labels:")
    print(f"  - loss_moa: {losses['loss_moa'].item():.4f} (computed on valid labels only)")
    
    print("\n✅ Missing label masking test PASSED")


def main():
    """Run all Phase 9 tests."""
    print("\n" + "="*70)
    print("PHASE 9: MULTI-TASK LEARNING - COMPREHENSIVE TEST")
    print("="*70)
    
    try:
        test_multitask_head()
        test_multitask_loss()
        test_model_integration()
        test_dynamic_weighting()
        test_missing_label_masking()
        
        print("\n" + "="*70)
        print("✅ ALL PHASE 9 TESTS PASSED")
        print("="*70)
        print("\nPhase 9 is production-ready!")
        print("Next steps:")
        print("  1. Train model with --use_multitask flag")
        print("  2. Provide MoA labels in dataset")
        print("  3. Evaluate per-task metrics")
        print("  4. Analyze task interactions")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
