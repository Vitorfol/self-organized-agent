#!/usr/bin/env python3
"""
Simple test to verify model detection functions work correctly.
"""

import sys
import os

# Add the programming_runs directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'programming_runs'))

# Import only the detection functions
from generators.model import is_gpt5_model, is_gpt4_or_older


def test_model_detection():
    """Test that model detection functions work correctly."""
    print("Testing model detection functions...")
    
    # GPT-5 models
    gpt5_models = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-turbo",
        "GPT-5-MINI-2025-08-07",
        "gpt5-mini",
    ]
    
    for model in gpt5_models:
        assert is_gpt5_model(model), f"Failed to detect {model} as GPT-5"
        assert not is_gpt4_or_older(model), f"Incorrectly detected {model} as GPT-4 or older"
        print(f"✓ {model} correctly detected as GPT-5")
    
    # GPT-4 and older models
    gpt4_older_models = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-0125-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "text-davinci-003",
    ]
    
    for model in gpt4_older_models:
        assert not is_gpt5_model(model), f"Incorrectly detected {model} as GPT-5"
        assert is_gpt4_or_older(model), f"Failed to detect {model} as GPT-4 or older"
        print(f"✓ {model} correctly detected as GPT-4/older")
    
    print("\n✅ All model detection tests passed!\n")


if __name__ == "__main__":
    try:
        test_model_detection()
        print("SUCCESS: Model compatibility layer is working correctly!")
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
