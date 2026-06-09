#!/usr/bin/env python
"""Test if all required modules can be imported."""
import sys

print("Testing imports...")

try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    from flask import Flask
    print("✓ Flask imported successfully")
except Exception as e:
    print(f"✗ Flask import failed: {e}")
    sys.exit(1)

try:
    from src.predict import load_model, load_image
    print("✓ src.predict imported successfully")
except Exception as e:
    print(f"✗ src.predict import failed: {e}")
    sys.exit(1)

print("\n✓ All imports successful! Flask app should work.")
print("Navigate to http://localhost:5000 to use the web interface.")
