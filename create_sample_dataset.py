#!/usr/bin/env python
"""Generate sample pneumonia dataset for testing."""
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Create dataset directory structure
dataset_dir = Path('sample_dataset')
for split in ['train', 'val']:
    for category in ['normal', 'pneumonia']:
        (dataset_dir / split / category).mkdir(parents=True, exist_ok=True)

# Generate sample images
def create_sample_image(filename, is_pneumonia=False):
    """Create a synthetic X-ray image."""
    # Create a base grayscale image
    img_array = np.random.randint(50, 150, (224, 224), dtype=np.uint8)
    
    if is_pneumonia:
        # Add some "pneumonia-like" artifacts (white spots)
        y, x = np.ogrid[:224, :224]
        center_y, center_x = 112, 112
        mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
        img_array[mask] = np.minimum(img_array[mask] + 100, 255)
    
    img = Image.fromarray(img_array, mode='L')
    img.save(filename)

# Generate training samples
print("Generating sample dataset...")
for i in range(30):
    create_sample_image(dataset_dir / 'train' / 'normal' / f'normal_{i:03d}.png', is_pneumonia=False)
    create_sample_image(dataset_dir / 'train' / 'pneumonia' / f'pneumonia_{i:03d}.png', is_pneumonia=True)

# Generate validation samples
for i in range(10):
    create_sample_image(dataset_dir / 'val' / 'normal' / f'normal_{i:03d}.png', is_pneumonia=False)
    create_sample_image(dataset_dir / 'val' / 'pneumonia' / f'pneumonia_{i:03d}.png', is_pneumonia=True)

print("✓ Sample dataset created at: sample_dataset/")
print("\nDataset structure:")
print("sample_dataset/")
print("├── train/")
print("│   ├── normal/        (30 images)")
print("│   └── pneumonia/     (30 images)")
print("└── val/")
print("    ├── normal/        (10 images)")
print("    └── pneumonia/     (10 images)")
print("\nNow run: python src/train.py --data-dir sample_dataset --output-model model/pneumonia_detector.h5")
