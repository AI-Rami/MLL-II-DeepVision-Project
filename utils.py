"""
utils.py
---------
Small helper file used across the project.

Main purpose:
- Detect whether a CUDA GPU is available.
- Return the correct device object (GPU if available, otherwise CPU).

This keeps the device selection logic clean and consistent across
all training scripts (main_cnn.py, main_transfer.py).
"""

import torch

def get_device() -> torch.device:
    """Pick CUDA GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
