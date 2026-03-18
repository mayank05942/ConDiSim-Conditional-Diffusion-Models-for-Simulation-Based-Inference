import torch
import torch.serialization
from pathlib import Path
import pyro.distributions.torch

# Add the required class to the safe globals list
torch.serialization.add_safe_globals([pyro.distributions.torch.MixtureSameFamily])

# Path to the GMM file
gmm_path = Path("/mimer/NOBACKUP/groups/ulio_inverse/mayank/probvlm_env/lib/python3.11/site-packages/sbibm/tasks/slcp/files/gmm.torch")

# Load the GMM with weights_only=False
gmm = torch.load(gmm_path, weights_only=False)

# Save it back in a compatible format
torch.save(gmm, gmm_path)

print(f"Successfully fixed the GMM file at {gmm_path}")
