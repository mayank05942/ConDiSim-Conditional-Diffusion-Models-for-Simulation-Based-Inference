import os
import numpy as np
import torch
import argparse
import time
from tqdm import tqdm
from vilar_model_architecture import DiffusionModel
from noise_scheduler import NoiseScheduler
from sampling import sample_posterior, save_posterior_samples

parser = argparse.ArgumentParser(description='Sample from trained diffusion model for Vilar parameter inference')
parser.add_argument('--budget', type=int, default=10000, help='Dataset budget (10000, 20000, or 30000)')
parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of posterior samples to generate')
parser.add_argument('--lambda_guidance', type=float, default=0.1, help='Guidance strength for classifier-free guidance')
parser.add_argument('--save_dir', type=str, default='posterior_samples', help='Directory to save samples')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

os.makedirs(args.save_dir, exist_ok=True)

device = torch.device(args.device)
print(f"Using device: {device}")

def load_reference_data(budget):
    dataset_path = f'vilar_dataset_{budget}.npz'
    print(f"Loading reference data: {dataset_path}")
    
    data = np.load(dataset_path, allow_pickle=True)
    
    true_theta = data['true_theta']
    true_ts_scaled = data['true_ts_scaled']
    theta_scaler = data['theta_scaler']
    
    true_theta = torch.tensor(true_theta, dtype=torch.float32)
    true_ts_scaled = torch.tensor(true_ts_scaled, dtype=torch.float32)
    
    print(f"True theta shape: {true_theta.shape}")
    print(f"True time series (scaled) shape: {true_ts_scaled.shape}")
    
    return true_theta, true_ts_scaled, theta_scaler

true_theta, true_ts_scaled, theta_scaler = load_reference_data(args.budget)

if args.model_path is None:
    args.model_path = os.path.join('models', f"vilar_diffusion_budget{args.budget}_best.pt")

print(f"Loading model from {args.model_path}")
checkpoint = torch.load(args.model_path, map_location=device)
model_args = checkpoint['args']

model = DiffusionModel(
    theta_dim=15,
    y_channels=3,
    y_seq_length=200,
    cfg_dropout_prob=0.2
).to(device)

noise_scheduler = NoiseScheduler(
    num_timesteps=model_args['num_timesteps'],
    beta_schedule=model_args['beta_schedule'],
    device=device
)

noise_scheduler.register_to_model(model)

filtered_state_dict = {}
for key, value in checkpoint['model_state_dict'].items():
    if key not in ['beta', 'alpha', 'alpha_hat', 'sqrt_alpha', 'sqrt_one_minus_alpha_bar', 'posterior_std']:
        filtered_state_dict[key] = value

model.load_state_dict(filtered_state_dict, strict=False)
model.eval()

print(f"Generating {args.num_samples} posterior samples...")
true_ts_scaled_device = true_ts_scaled.to(device).unsqueeze(0)

start_time = time.time()

posterior_samples = sample_posterior(
    model=model,
    y_observed=true_ts_scaled_device,
    num_samples=args.num_samples,
    device=device,
    lambda_guidance=args.lambda_guidance
)

end_time = time.time()
sampling_time = end_time - start_time
print(f"Sampling completed in {sampling_time:.2f} seconds")

dataset_path = f'vilar_dataset_{args.budget}.npz'
print(f"Loading dataset from {dataset_path} to get theta_scaler")
data = np.load(dataset_path, allow_pickle=True)

if 'theta_scaler' in data:
    theta_scaler = data['theta_scaler'].item()
    print("Found theta_scaler in dataset, using it for denormalization")
    
    posterior_samples = theta_scaler.inverse_transform(posterior_samples.cpu().numpy())
else:
    print("ERROR: theta_scaler not found in dataset. Cannot properly denormalize samples.")
    print("Using raw samples without denormalization.")
    posterior_samples = posterior_samples.cpu().numpy()

print(f"Parameter ranges after processing:")
print(f"  Min: {np.min(posterior_samples)}")
print(f"  Max: {np.max(posterior_samples)}")
print(f"  Mean: {np.mean(posterior_samples)}")


arrays_to_save = {
    "theta_samples": posterior_samples,
    "true_parameters": true_theta.cpu().numpy(),
    "observed_data": true_ts_scaled.cpu().numpy(),
    "lambda_guidance": args.lambda_guidance,
    "budget": args.budget
}

save_path = save_posterior_samples(
    samples=arrays_to_save,
    save_dir=args.save_dir,
    filename=f"vilar_posterior_samples_budget{args.budget}_lambda{args.lambda_guidance}.npz"
)

print("Sampling complete!")
