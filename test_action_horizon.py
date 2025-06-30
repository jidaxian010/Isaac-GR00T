#!/usr/bin/env python3
"""
Test script to verify the action horizon of your trained model.
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gr00t.model.gr00t_n1 import GR00T_N1
from transformers.feature_extraction_utils import BatchFeature

def test_action_horizon():
    """Test if the model outputs 4 timesteps instead of 16."""
    
    # Load your trained model
    model_path = "./checkpoints/checkpoint-300"  # Update this path
    print(f"Loading model from: {model_path}")
    
    model = GR00T_N1.from_pretrained(model_path)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    device = next(model.parameters()).device
    
    # Dummy backbone output (vision + language features)
    backbone_features = torch.randn(batch_size, 17, 1024, device=device)  # 17 timesteps from DiT
    backbone_attention_mask = torch.ones(batch_size, 17, device=device)
    backbone_output = BatchFeature(data={
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask
    })
    
    # Dummy action input
    state = torch.randn(batch_size, 32, device=device)  # 32-dim state
    action = torch.randn(batch_size, 4, 7, device=device)  # 4 timesteps, 7-dim actions
    action_mask = torch.ones(batch_size, 4, 7, device=device)
    embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    action_input = BatchFeature(data={
        "state": state,
        "action": action,
        "action_mask": action_mask,
        "embodiment_id": embodiment_id
    })
    
    print("\n" + "="*50)
    print("TESTING ACTION HORIZON")
    print("="*50)
    
    # Test 1: Check model configuration
    print(f"Model config action_horizon: {model.config.action_horizon}")
    print(f"Action head action_horizon: {model.action_head.action_horizon}")
    
    # Test 2: Run forward pass (training mode)
    print("\n--- Training Forward Pass ---")
    with torch.no_grad():
        output = model.action_head.forward(backbone_output, action_input)
        print(f"Training loss: {output.data['loss']}")
    
    # Test 3: Run inference (get_action)
    print("\n--- Inference (get_action) ---")
    with torch.no_grad():
        action_pred = model.action_head.get_action(backbone_output, action_input)
        predicted_actions = action_pred.data["action_pred"]
        print(f"Predicted actions shape: {predicted_actions.shape}")
        print(f"Expected shape: (batch_size, 4, 7)")
        
        if predicted_actions.shape[1] == 4:
            print("✅ SUCCESS: Model outputs 4 timesteps!")
        else:
            print(f"❌ FAILURE: Model outputs {predicted_actions.shape[1]} timesteps, expected 4")
    
    # Test 4: Check if using adaptive decoder
    print("\n--- Decoder Check ---")
    if hasattr(model.action_head, 'action_decoder_adaptive'):
        print("✅ Using CategorySpecificSequenceMLP")
        print(f"Target sequence length: {model.action_head.action_decoder_adaptive.target_seq_length}")
        print(f"CategorySpecificSequenceMLP type: {type(model.action_head.action_decoder_adaptive)}")
        
        # Check the parameters
        adaptive_decoder = model.action_head.action_decoder_adaptive
        print(f"Layer 1 W shape: {adaptive_decoder.layer1.W.shape}")
        print(f"Layer 1 b shape: {adaptive_decoder.layer1.b.shape}")
        print(f"Layer 2 W shape: {adaptive_decoder.layer2.W.shape}")
        print(f"Layer 2 b shape: {adaptive_decoder.layer2.b.shape}")
    else:
        print("❌ Not using CategorySpecificSequenceMLP")
    
    # Test 5: Check model configuration
    print("\n--- Model Configuration ---")
    print(f"Model config action_horizon: {model.config.action_horizon}")
    print(f"Action head action_horizon: {model.action_head.action_horizon}")
    print(f"Action head action_dim: {model.action_head.action_dim}")
    print(f"Action head config action_dim: {model.action_head.config.action_dim}")
    
    # Test 6: Test the CategorySpecificSequenceMLP directly
    print("\n--- Direct CategorySpecificSequenceMLP Test ---")
    if hasattr(model.action_head, 'action_decoder_adaptive'):
        adaptive_decoder = model.action_head.action_decoder_adaptive
        
        # Create dummy input
        dummy_model_output = torch.randn(1, 17, 1024, device=device)
        dummy_embodiment_id = torch.zeros(1, dtype=torch.long, device=device)
        
        with torch.no_grad():
            output = adaptive_decoder(dummy_model_output, dummy_embodiment_id)
            print(f"Direct CategorySpecificSequenceMLP output shape: {output.shape}")
            print(f"Expected shape: (1, 4, 7)")
            
            if output.shape[1] == 4:
                print("✅ CategorySpecificSequenceMLP outputs 4 timesteps correctly!")
            else:
                print(f"❌ CategorySpecificSequenceMLP outputs {output.shape[1]} timesteps, expected 4")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_action_horizon() 