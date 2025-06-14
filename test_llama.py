#!/usr/bin/env python3
"""Test script for Llama 3 8B model loading."""

import torch
import sys
import os

# Add the attention_viz package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'attention_viz'))

from attention_viz.utils.helpers import load_model_and_tokenizer, get_model_info

def test_llama_loading():
    """Test loading Llama 3 8B model with specified parameters."""
    
    # Common Llama 3 8B model names to try
    model_names = [
        "meta-llama/Llama-3-8B",
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]
    
    print("Testing Llama 3 8B model loading...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    for model_name in model_names:
        print(f"\nTrying to load: {model_name}")
        try:
            # Test loading the model
            model, tokenizer = load_model_and_tokenizer(
                model_name=model_name,
                output_attentions=True,
                device="cpu"  # Use CPU for testing to avoid GPU memory issues
            )
            
            print(f"✅ Successfully loaded {model_name}")
            
            # Get model info
            model_info = get_model_info(model)
            print("Model Info:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            
            # Test tokenization
            test_text = "Hello, how are you today?"
            inputs = tokenizer(test_text, return_tensors="pt")
            print(f"✅ Tokenization successful. Input shape: {inputs['input_ids'].shape}")
            
            # Test forward pass (without gradients to save memory)
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    print(f"✅ Forward pass successful. Got attention weights: {len(outputs.attentions)} layers")
                    print(f"   First layer attention shape: {outputs.attentions[0].shape}")
                else:
                    print("⚠️  Forward pass successful but no attention weights returned")
            
            # Clean up to free memory
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True  # Success with this model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            continue
    
    print("\n❌ Failed to load any Llama 3 8B model variant")
    return False

if __name__ == "__main__":
    success = test_llama_loading()
    if success:
        print("\n🎉 Llama 3 8B model loading test completed successfully!")
    else:
        print("\n💥 Llama 3 8B model loading test failed.")
        print("Make sure you have:")
        print("1. Hugging Face transformers library installed")
        print("2. Access to the Llama models (may require authentication)")
        print("3. Sufficient system memory") 