import torch
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch import Tensor
from typing import Self, List, Callable
from nfdm.model.model_abc import GenerativeMethod
from timm.utils.model_ema import ModelEmaV3
from tqdm import tqdm
from torch.amp import GradScaler
from line_profiler import profile

# Simple MLP for 2D data instead of DDPM (which is for images)
class Simple2DMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 2, hidden_dim: int = 512, time_embed_dim: int = 256):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(in_dim + time_embed_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # x: (batch_size, 2)
        # t: (batch_size, 1, 1, 1) or (batch_size, 1) - need to handle both
        
        if t.dim() > 2:
            t = t.view(t.shape[0], -1)  # Flatten to (batch_size, 1)
        
        t_embed = self.time_embed(t)  # (batch_size, time_embed_dim)
        
        # Concatenate input and time embedding
        x_t = torch.cat([x, t_embed], dim=1)  # (batch_size, in_dim + time_embed_dim)
        
        return self.net(x_t)

class Var_Scheduler(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.sp = nn.Softplus()
        
    def forward(self, t: Tensor) -> Tensor:
        if t.dim() > 2:
            t = t.view(t.shape[0], -1)
        return self.sp(self.net(t)) + 1e-3

class AffineNeural(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.net = Simple2DMLP(in_dim=in_dim, out_dim=out_dim).to(device)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        m_s = self.net(x, t)
        m, s = m_s.chunk(2, dim=1)

        # Handle different tensor shapes for 2D data
        if t.dim() > 2:
            t = t.view(t.shape[0], -1)
        t = t.view(-1, 1)  # (batch_size, 1)
        
        m = (1 - t) * x + t * (1 - t) * m
        s = (0.01 ** (1-t)) * ((5 * torch.clamp(torch.sigmoid(s),min=1e-4)) ** (t * (1-t)))

        return m, s

def jvp(f, x: Tensor, v: Tensor) -> tuple[Tensor, ...]:
    return torch.autograd.functional.jvp(
        f, x, v, 
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f, t: Tensor) -> tuple[Tensor, ...]:
    return jvp(f, t, torch.ones_like(t))

class AffineTransform(nn.Module):
    def __init__(self, flow: AffineNeural):
        super().__init__()
        self.flow = flow

    def get_t_dir(self, x: Tensor, t: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        def f(x_in):
            def f_(t_in):
                return self.flow(x_in, t_in)
            return f_

        return t_dir(f(x), t)

    def forward(self, eps: Tensor, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (m, s), (dm, ds) = self.get_t_dir(x, t)

        z = m + s * eps
        dz = dm + ds * eps
        score = - eps / s

        return z, dz, score

    def inverse(self, z: Tensor, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        (m, s), (dm, ds) = self.get_t_dir(x, t)

        eps = (z - m) / s
        dz = dm + ds / s * (z - m)
        score = (m - z) / s ** 2

        return eps, dz, score

def score_based_sde_drift(dz: Tensor, score: Tensor, g2: Tensor) -> Tensor:
    return dz - 0.5 * g2 * score

class NeuralDiffusion(nn.Module):
    def __init__(self, transform: AffineTransform, pred: Simple2DMLP, vol: nn.Module, curvature_wt: float = 0.0):
        super().__init__()
        self.transform = transform
        self.pred = pred
        self.vol = vol
        self.curvature_wt = curvature_wt
    def compute_curvature_penalty(self, eps: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared second derivative of the flow trajectory w.r.t. time
        to encourage linearity in time (optimal transport regularization).
        """
        def second_order_jvp(f, t_):
            return torch.autograd.functional.jvp(
                lambda t__: torch.autograd.functional.jvp(f, t__, torch.ones_like(t__), create_graph=True)[1],
                t_, torch.ones_like(t_),
                create_graph=True
            )[1]

        z_fn = lambda t_: self.transform(eps, t_, x)[0]  # z = m + s * eps
        z_tt = second_order_jvp(z_fn, t)
        return torch.mean(z_tt ** 2)

    def forward(self, x: Tensor, t: Tensor):
        eps = torch.randn_like(x)
        z, f_dz, f_score = self.transform(eps, t, x)
        x_ = self.pred(z, t)
        _, r_dz, r_score = self.transform.inverse(z, t, x_)

        g2 = (self.vol(t) ** 2)
        if g2.dim() > 2:
            g2 = g2.view(-1, 1)
        
        f_drift = score_based_sde_drift(f_dz, f_score, g2)
        r_drift = score_based_sde_drift(r_dz, r_score, g2)

        loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        # if(self.curvature_wt > 0):
        #     curvature_penalty = self.compute_curvature_penalty(eps, t, x)
        #     loss += self.curvature_wt * curvature_penalty
        return loss

@torch.no_grad()
def solve_sde(
        sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        show_pbar: bool=False,
        device = 'cpu'
):
    bs = z.shape[0]
    t_steps = torch.linspace(ts, tf, n_steps + 1, device=device)
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    path = [z]
    pbar = tqdm if show_pbar else (lambda a: a)
    for t in pbar(t_steps[:-1]):
        t_expanded = t.expand(bs, 1)
        f, g = sde(z, t_expanded)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        path.append(z)

    return z, (t_steps, torch.stack(path))

class NFDM_2D(GenerativeMethod):
    def __init__(
        self: Self, 
        in_dim: int = 2, 
        batch_size: int = 128, 
        lr: float = 2e-4,
        device: str = 'cpu',
        gamma: float = 0.99, 
        amp: bool = True, 
        curvature_weight: float = 0.0
    ) -> None:
        super(NFDM_2D, self).__init__()
        self.curvature_weight = curvature_weight
        self.transform = AffineTransform(flow=AffineNeural(in_dim, 2*in_dim, device))
        self.predictor = Simple2DMLP(in_dim=in_dim, out_dim=in_dim).to(device)
        self.scheduler = Var_Scheduler().to(device)
        self.model = NeuralDiffusion(self.transform, self.predictor, self.scheduler,self.curvature_weight).to(device)

        self.ema = ModelEmaV3(self.model, device=device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.amp = False
        self.lr_sched = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    @profile
    def train(
        self: Self, 
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 100, 
        checkpoint: bool = False, 
        ckpt_name: str = 'checkpoint_2d'
    ) -> List[float]: 
        if checkpoint:
            try: 
                print(os.getcwd())
                #exit()
                self.load(ckpt_name)
                print("Checkpoint loaded")
            except FileNotFoundError:
                print('No checkpoint found')
                
        scaler = GradScaler(self.device, enabled=self.amp)
        max_loss = float('inf')
        self.it = 0 
        self.epoch_loss = 0 
        
        for epoch in (pbar := tqdm(range(epochs))): 
            batch_loss = 0 
            lr = self.lr_sched.get_last_lr()[0]
            
            for data in train_loader:
                self.it += 1
                # For toy data, data is just the points (no labels)
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    points, _ = data  # If data has labels
                else:
                    points = data  # If data is just points
                
                points = points.to(self.device)
                batch_size = points.shape[0]
                timesteps = torch.rand((batch_size, 1), device=self.device)
                
                outs = self.model(points, timesteps)
                loss = torch.mean(outs)    
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.ema.update(self.model)
        
                loss_item = loss.detach().item()
                batch_loss += loss_item
                
                pbar.set_description(f"Training NFDM_2D | Batch Loss: {loss_item:.3f} | Last Loss: {self.epoch_loss:.3f} | LR: {lr:.7f} | Iter: {self.it}")

            self.lr_sched.step()
            self.epoch_loss = batch_loss / len(train_loader)
                
            if checkpoint and batch_loss < max_loss: 
                self.save('checkpoint_2d')
                max_loss = batch_loss

    def save(
        self: Self,
        dir: str = 'nfdm_2d'
    ) -> None: 
        #print("In save model function:", os.getcwd(), os.listdir('.'))
        torch.save({
            'nfdm' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'model/{dir}.pt')
        
    def load(
        self: Self, 
        dir: str = 'nfdm_2d'
    ) -> None: 
        print("In load model function:", os.getcwd(), os.listdir('.'))
        #load_obj = torch.load(f'model/{dir}.pt', weights_only=True)
        #print("Files in cuttent directory:", os.listdir('.'))
        ckpt_path = f'model/{dir}.pt'
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found at {os.path.abspath(ckpt_path)}.")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
    
        load_obj = torch.load(ckpt_path, map_location=self.device)
        print(f"Loading model from {dir}.pt", load_obj==None)
        self.model.load_state_dict(load_obj['nfdm'])
        self.ema.load_state_dict(load_obj['ema'])
        self.optimizer.load_state_dict(load_obj['optimizer'])

    @torch.no_grad()
    def generate(
        self: Self,
        num_samples: int = 1000,
        num_timesteps: int = 300, 
        xlim: tuple = (-10, 80),
        ylim: tuple = (-30, 30),
    ) -> None: 
        self.model.eval()
        self.ema.module.eval()

        def sde(z_in, t_in):
            x_ = self.predictor(z_in, t_in)
            _, dz, score = self.transform.inverse(z_in, t_in, x_)
            g = self.scheduler(t_in)
            g2 = g ** 2
            drift = score_based_sde_drift(dz, score, g2)
            return drift, g

        # Start with random 2D points
        z = torch.randn((num_samples, 2), device=self.device)
        x, (ts, zs) = solve_sde(sde, z, 1, 0, num_timesteps, show_pbar=True, device=self.device)
        
        # Plot the generation process
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        # Show evolution at different timesteps
        timestep_indices = np.linspace(0, len(zs)-1, 10).astype(int)
        
        for i, idx in enumerate(timestep_indices):
            points = zs[idx].cpu().numpy()
            axes[i].scatter(points[:, 0], points[:, 1], s=1, alpha=0.6)
            axes[i].set_title(f't = {ts[idx]:.2f}')
            axes[i].set_xlim(xlim)
            axes[i].set_ylim(ylim)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('nfdm_2d_generation.png', dpi=150)
        plt.show()
        
        # Plot final result
        plt.figure(figsize=(10, 8))
        final_points = x.cpu().numpy()
        plt.scatter(final_points[:, 0], final_points[:, 1], s=2, alpha=0.7)
        plt.title(f'Generated 2D Points ({num_samples} samples)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig('nfdm_2d_final.png', dpi=150)
        plt.show()
        
        self.model.train()
        self.ema.module.train()

# Usage example:
if __name__ == "__main__": 
    # Test with random 2D data
    x = torch.randn((10, 2))
    model = NFDM_2D(in_dim=2)
    t = torch.rand((10, 1))
    out = model.model(x, t)
    print(f"Output shape: {out.shape}")