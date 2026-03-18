import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import sbibm
import copy
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import argparse
from utils import load_data, create_dataloaders, noise_schedule, set_seed
from bayesflow import benchmarks
from torch.nn.utils.parametrizations import orthogonal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimeEmbeddingModule(nn.Module):
    def __init__(self, code_dim: int = 64, sin_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.sin_dim = sin_dim
        self.code_dim = code_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.sin_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )
        self.initialize()

    def initialize(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, t):
        t_emb = timestep_embedding(t, self.sin_dim)
        return self.mlp(t_emb)


class YFeatureEncoder(nn.Module):
    def __init__(self, y_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
        )
        self.initialize()
    
    def initialize(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, y):
        return self.net(y)


class TimedepedentFILM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        self.initialize()
 
    def initialize(self):
        last_linear = None
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, c):
        gb = self.mlp(c)
        return gb.chunk(2, dim=-1)


class ReverseDiffusionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        
        self.post_film_mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        self.initialize()

    def initialize(self):
        for module in self.post_film_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        if isinstance(self.post_film_mlp[-1], nn.Linear):
            nn.init.zeros_(self.post_film_mlp[-1].weight)
            nn.init.zeros_(self.post_film_mlp[-1].bias)

    def apply_film(self, x, gamma, beta):
        return x * (1 + gamma) + beta

    def forward(self, x, gamma, beta):
        residual = x
        h = self.norm(x)
        h = self.apply_film(h, gamma, beta)
        h = self.post_film_mlp(h)
        return h + residual
    


class ReverseDiffusionModel(nn.Module):
    def __init__(self, theta_dim, y_dim):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.num_timesteps = 0

        self.timembedding = TimeEmbeddingModule(64, 64, 256)
        self.y_enc = YFeatureEncoder(y_dim)

        self.theta_head = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.GELU(),
            nn.Linear(128,128),
            nn.LayerNorm(128),
        )

        self.film_per_block = nn.ModuleList([
            TimedepedentFILM()
            for _ in range(5)
        ])

        self.theta_conditioned_blocks = nn.ModuleList([
            ReverseDiffusionBlock()
            for _ in range(5)
        ])

        self.out_norm = nn.LayerNorm(128)
        self.theta_tail = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, theta_dim),
        )
        
        self.initialize()

    def initialize(self):
        for module in self.theta_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        last_linear = None
        for module in self.theta_tail.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, theta_t, y, t):
        t_code = self.timembedding(t)
        y_code = self.y_enc(y)
        c = torch.cat([y_code, t_code], dim=-1)
        theta_projected = self.theta_head(theta_t)
        for i, blk in enumerate(self.theta_conditioned_blocks):
            gamma, beta = self.film_per_block[i](c)
            theta_projected = blk(theta_projected, gamma, beta)

        theta_projected = self.out_norm(theta_projected)
        return self.theta_tail(theta_projected)
    

    def set_noise_schedule(self, beta: torch.Tensor, eps: float = 1e-12):
        """Register noise schedule and precompute constants as model buffers.
        Expects beta shape [T]. Computes alpha, alpha_hat, sqrt terms, and posterior_std.
        """
        with torch.no_grad():
            beta = beta.detach()
            self.register_buffer('beta', beta)
            alpha = 1.0 - beta
            self.register_buffer('alpha', alpha)
            alpha_hat = torch.cumprod(alpha, dim=0)
            self.register_buffer('alpha_hat', alpha_hat)
            self.num_timesteps = beta.numel()

            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=eps))
            sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_hat, min=eps))
            self.register_buffer('sqrt_alpha', sqrt_alpha)
            self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)

            alpha_hat_tm1 = torch.cat([
                torch.ones(1, device=alpha_hat.device, dtype=alpha_hat.dtype),
                alpha_hat[:-1]
            ], dim=0)
            posterior_var = beta * (1.0 - alpha_hat_tm1) / torch.clamp(1.0 - alpha_hat, min=eps)
            posterior_var = torch.clamp(posterior_var, min=eps)
            posterior_std = torch.sqrt(posterior_var)
            self.register_buffer('posterior_std', posterior_std)


    def sample(self, N, y_observed, device=None):
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            theta = torch.randn((N, self.theta_dim), device=device)
            T = self.num_timesteps

            for t in reversed(range(T)):
                t_tensor = torch.full((N,), t, device=device)
                beta_t = self.beta[t].unsqueeze(-1)
                sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
                sqrt_alpha_t = self.sqrt_alpha[t].unsqueeze(-1)

                eps_pred = self.forward(theta, y_observed, t_tensor)

                mean = (theta - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_pred) / sqrt_alpha_t

                if t > 0:
                    noise =  torch.randn_like(theta)
                    posterior_std_t = self.posterior_std[t].unsqueeze(-1)
                    theta = mean + posterior_std_t * noise
                else:
                    theta = mean
        return theta


def diffusion_loss(model,
                   theta_0,
                   y,
                   num_timesteps,
                   alpha_hat,
                   use_snr_weighting: bool = True,
                   eps: float = 1e-12,
                   loss_type: str = "huber",
                   huber_beta: float = 1.0):
    """
    Compute diffusion loss (predicting noise) with optional Min-SNR gamma weighting.

    loss_type: "mse" (default) or "huber" (smooth L1 with beta=huber_beta).
    If True, applies weights w_t = min(SNR_t, gamma) / SNR_t with fixed gamma=5.0 (Section 3.4 of
    https://arxiv.org/abs/2303.09556, adapted for epsilon prediction).
    """
    device = theta_0.device
    t = torch.randint(0, num_timesteps, (theta_0.size(0),), device=device).long()
    noise = torch.randn_like(theta_0)

    abar_t = alpha_hat[t]
    theta_t = (torch.sqrt(abar_t.unsqueeze(-1)) * theta_0 + 
               torch.sqrt(torch.clamp(1 - abar_t, min=eps)).unsqueeze(-1) * noise)

    pred_noise = model(theta_t, y, t)

    if not use_snr_weighting:
        if loss_type == "mse":
            return F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
        elif loss_type == "huber":
            return F.smooth_l1_loss(pred_noise.float(), noise.float(), beta=huber_beta, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    if loss_type == "mse":
        loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="none")
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(pred_noise.float(), noise.float(), beta=huber_beta, reduction='none')
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    loss = loss.mean(dim=1)



    snr = torch.clamp(abar_t, min=eps) / torch.clamp(1.0 - abar_t, min=eps)
    gamma = 5.0
    base_weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    loss = (loss * base_weight).mean()
    return loss

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                num_epochs,
                num_timesteps,
                alpha_hat,
                patience=12,
                min_epochs_before_es=50,
                use_snr_weighting: bool = False,
                device=None):
    if device is None:
        device = next(model.parameters()).device
    
    best_val_loss = float('inf')
    best_model = None
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = diffusion_loss(model, theta_batch, y_batch, num_timesteps, alpha_hat, use_snr_weighting=use_snr_weighting)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)
                val_loss += diffusion_loss(model, theta_batch, y_batch, num_timesteps, alpha_hat, use_snr_weighting=use_snr_weighting).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                model = best_model
                break
    
    return model, train_losses, val_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta_dim", type=int, default=5, help="Dimension of theta")
    parser.add_argument("--y_dim", type=int, default=10, help="Dimension of y")
    parser.add_argument("--num_timesteps", type=int, default=500, help="Number of timesteps for diffusion")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Starting value of beta for noise schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending value of beta for noise schedule")
    parser.add_argument("--patience", type=int, default=12, help="Patience for early stopping")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--beta_schedule', type=str, default='cosine', help="Beta schedule: 'linear', 'quadratic', or 'cosine'")
    parser.add_argument('--use_snr_weighting', action='store_true', help='Enable Min-SNR loss weighting with fixed gamma=5.0')
    parser.add_argument('--task_name', type=str, default='slcp_distractors', help='Task name for the experiment')
    parser.add_argument("--simulation_budgets", type=int, nargs='+', default=[10000, 20000, 30000], help="Simulation budgets to use")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs for each simulation budget")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Arguments: {args}")

    global hyperparams
    hyperparams = {
        'theta_dim': args.theta_dim,
        'y_dim': args.y_dim,
        'num_timesteps': args.num_timesteps,
        'learning_rate': args.learning_rate,
        'beta_start': args.beta_start,
        'beta_end': args.beta_end,
        'patience': args.patience,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'beta_schedule': args.beta_schedule,
        'use_snr_weighting': args.use_snr_weighting,
        'task_name': args.task_name,
        'simulation_budgets': args.simulation_budgets,
        'num_runs': args.num_runs,
    }


    simulation_budgets, num_runs = hyperparams['simulation_budgets'], hyperparams['num_runs']

    for budget in simulation_budgets:
        for run in range(1, num_runs + 1):
            task_name = hyperparams['task_name']

            if task_name in ["lotka_volterra", "sir"]:
                benchmark = sbibm.Benchmark(task_name)
                task = sbibm.get_task(task_name)

                def simulator(theta):
                    x = benchmark.generative_model.simulator(theta)
                    return x["sim_data"]

                def prior(N):
                    x = benchmark.generative_model.prior(N)
                    return x["prior_draws"]

                num_samples = hyperparams['num_samples']
                theta_samples = prior(num_samples)
                y_data = simulator(theta_samples)

                theta_scaler = StandardScaler().fit(theta_samples)
                y_scaler = StandardScaler().fit(y_data)
                thetas = torch.tensor(theta_scaler.transform(theta_samples), dtype=torch.float32)
                y_data = torch.tensor(y_scaler.transform(y_data), dtype=torch.float32)

                dataset = TensorDataset(thetas, y_data)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

            else:
                thetas, y_data, theta_scaler, y_scaler, task = load_data(task_name, budget)
                train_loader, val_loader = create_dataloaders(thetas, y_data, hyperparams['batch_size'])

            beta = noise_schedule(hyperparams['num_timesteps'],
                                  hyperparams['beta_start'],
                                  hyperparams['beta_end'],
                                  hyperparams['beta_schedule'],
                                  max_beta=0.02).to(device)
            print(f"Noise schedule: {hyperparams['beta_schedule']}, T={beta.numel()}, min={beta.min().item():.6f}, max={beta.max().item():.6f}, sum={beta.sum().item():.4f}")

            model = ReverseDiffusionModel(
                hyperparams['theta_dim'],
                hyperparams['y_dim']
            ).to(device)
            
            model.set_noise_schedule(beta)
            print(f"alpha_hat[0]={model.alpha_hat[0].item():.6f}, alpha_hat[-1]={model.alpha_hat[-1].item():.6f}")
            alpha_hat = model.alpha_hat
            
            print("Model Architecture:")
            print(model)
            print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


            optimizer = optim.AdamW(model.parameters(),
                                    lr=hyperparams['learning_rate'])
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             factor=0.5,
                                                             patience=20,
                                                             min_lr=1e-6)

            start_time = time.time()
            model, train_losses, val_losses = train_model(model,
                                                         train_loader,
                                                         val_loader, 
                                                         optimizer,
                                                         scheduler, 
                                                         hyperparams['num_epochs'], 
                                                         hyperparams['num_timesteps'], 
                                                         alpha_hat,
                                                         hyperparams['patience'],
                                                         min_epochs_before_es=20,
                                                         use_snr_weighting=hyperparams['use_snr_weighting'])
            end_time = time.time()
            training_time = end_time - start_time

            print(f"Training completed in {training_time/3600:.2f} hours ({training_time:.2f} seconds)")

            results_dir = os.path.join("results", task_name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            training_info = {
                'model_state_dict': model.state_dict(),
                'training_time': training_time,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            model_filename = f"{task_name}_budget_{budget}_run_{run}.pth"
            model_path = os.path.join(results_dir, model_filename)
            torch.save(training_info, model_path)
            print(f"Model saved to {model_path}")
            
            training_time_minutes = training_time / 60
            
            time_log_path = os.path.join(results_dir, "training_times.txt")
            with open(time_log_path, 'a') as f:
                f.write(f"Task: {task_name}\n")
                f.write(f"Simulation Budget: {budget}\n")
                f.write(f"Run: {run}\n")
                f.write(f"Training Time (minutes): {training_time_minutes:.2f}\n")
                f.write("-" * 50 + "\n")
            
            print(f"Training time: {training_time_minutes/60:.2f} hours ({training_time_minutes:.2f} minutes)")

            model.eval()

            inference_time = 0.0
            
            if task_name == 'sir':
                loaded_data = np.load('sir_task_data.npz', allow_pickle=True)
                true_param_1 = loaded_data['set_1'].item()['true_param']
                observed_data_1 = loaded_data['set_1'].item()['observed_data'][0]
                observed_data_1 = observed_data_1.reshape(1, -1)
                observed_data_1 = y_scaler.transform(observed_data_1)
                observed_data_1 = torch.tensor(observed_data_1, dtype=torch.float32)
                y_observed = observed_data_1.repeat(hyperparams['num_samples'], 1).to(device)
                theta_samples = model.sample(hyperparams['num_samples'], y_observed)
                theta_samples_descaled = theta_scaler.inverse_transform(theta_samples.cpu().numpy())
            else:
                y_obs_np = np.asarray(task.get_observation(num_observation=1), dtype=np.float32).reshape(1, -1)
                y_observed = y_scaler.transform(y_obs_np)
                inference_start_time = time.time()
                y_observed = torch.tensor(y_observed, dtype=torch.float32).repeat(hyperparams['num_samples'], 1).to(device)
                theta_samples = model.sample(hyperparams['num_samples'], y_observed)
                inference_end_time = time.time()
                inference_time = inference_end_time - inference_start_time
                print(f"Inference time: {inference_time:.2f} seconds (or {inference_time/60:.2f} minutes)")
                print(f"Samples (scaled): mean={theta_samples.mean().item():.4f}, std={theta_samples.std().item():.4f}")
                theta_samples_descaled = theta_scaler.inverse_transform(theta_samples.cpu().numpy())
                print(f"Samples (descaled): mean={np.mean(theta_samples_descaled):.4f}, std={np.std(theta_samples_descaled):.4f}")
                
            with open(time_log_path, 'a') as f:
                inference_time_minutes = inference_time / 60
                f.write(f"Task: {task_name}\n")
                f.write(f"Simulation Budget: {budget}\n")
                f.write(f"Run: {run}\n")
                f.write(f"Inference Time (minutes): {inference_time_minutes:.2f}\n")
                f.write("-" * 50 + "\n")
            posterior_filename = f"{task_name}_run_{run}_budget_{budget}.npz"
            posterior_path = os.path.join(results_dir, posterior_filename)
            np.savez(posterior_path, theta_samples=theta_samples_descaled)
            print(f"Posterior samples saved to {posterior_path}")





if __name__ == "__main__":
    main()
