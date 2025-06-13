#!/usr/bin/env python3
"""
Test script to verify HF token authentication works
"""

import os
import sys
import traceback

# Import HF token configuration
try:
    from hf_config import HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
    print("✅ HF token loaded from hf_config.py")
except ImportError:
    print("❌ Failed to load HF token from hf_config.py")
    sys.exit(1)

# Test importing LLMInference
try:
    from llm_inference import LLMInference
    print("✅ LLMInference imported successfully - token authentication working!")
except Exception as e:
    print(f"❌ Failed to import LLMInference: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)

print("🎉 All authentication tests passed!") 