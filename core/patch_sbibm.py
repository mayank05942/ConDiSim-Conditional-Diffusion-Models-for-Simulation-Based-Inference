import os

import torch

try:
    import pyro
except ImportError:  # pragma: no cover
    pyro = None

import torch.serialization
import pyro.distributions.torch
from pathlib import Path

# Add the required class to the safe globals list
if pyro is not None and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([pyro.distributions.torch.MixtureSameFamily])

# Create a monkey patch for torch.load
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    # Force weights_only=False for compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

# Replace the original torch.load with our patched version
torch.load = patched_torch_load

print("Patched torch.load to use weights_only=False by default")
print("Added pyro.distributions.torch.MixtureSameFamily to safe globals")
