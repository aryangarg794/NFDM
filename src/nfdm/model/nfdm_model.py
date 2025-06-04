import torch
import numpy as np 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Self, List, Callable
from nfdm.model.model_abc import GenerativeMethod
from timm.utils.model_ema import ModelEmaV3
from tqdm import tqdm
from torch.amp import GradScaler
from line_profiler import profile
from nfdm.model.ddpm import DDPM

# class Continuous_DDPM(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, num_timesteps: int = 300):
#         super().__init__()

#         self.num_timesteps = num_timesteps
#         self.net = DDPM(in_channels=in_channels, out_channels=out_channels, horizon=num_timesteps+1)
    
#     def forward(self, x: Tensor, t: Tensor) -> Tensor:
#         # t_i = (t * self.num_timesteps).long()
#         return self.net(x, t)

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
        return self.sp(self.net(t)) + 1e-2
    
class AffineNeural(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()

        self.net = DDPM(in_channels=in_channels, out_channels=out_channels).to(device)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        m_ls = self.net(x, t)
        m, ls = m_ls.chunk(2, dim=1)

        t = t.view(-1, 1, 1, 1)
        m = (1 - t) * x + t * (1 - t) * m
        ls = (1 - t) * np.log(0.01) + t * (1 - t) * ls

        return m, torch.exp(ls)
    
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
    def __init__(self, transform: AffineTransform, pred: DDPM, vol: nn.Module):
        super().__init__()

        self.transform = transform
        self.pred = pred
        self.vol = vol

    def forward(self, x: Tensor, t: Tensor):
        eps = torch.randn_like(x)

        z, f_dz, f_score = self.transform(eps, t, x)

        x_ = self.pred(z, t)
        _, r_dz, r_score = self.transform.inverse(z, t, x_)

        g2 = (self.vol(t) ** 2).view(-1, 1, 1, 1)
        f_drift = score_based_sde_drift(f_dz, f_score, g2)
        r_drift = score_based_sde_drift(r_dz, r_score, g2)

        loss = 0.5 * (f_drift - r_drift) ** 2 / g2
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
        t = t.expand(bs, 1, 1, 1)

        f, g = sde(z, t)

        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2

        path.append(z)

    return z, (t_steps, torch.stack(path))

class NFDM(GenerativeMethod):
        
    def __init__(
        self: Self, 
        in_channels: int = 3, 
        batch_size: int = 128, 
        lr: float = 2e-4,
        device: str = 'cpu',
        gamma: float = 0.99, 
        amp: bool = True, 
    ) -> None:
        super(NFDM, self).__init__()
 
        self.transform = AffineTransform(flow=AffineNeural(in_channels, 2*in_channels, device))
        self.predictor = DDPM(in_channels=in_channels, out_channels=in_channels).to(device)
        self.scheduler = Var_Scheduler().to(device)
        self.model = NeuralDiffusion(self.transform, self.predictor, self.scheduler).to(device)

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
    ) -> List[float]: 
        if checkpoint:
            try: 
                self.load('checkpoint')
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
                images, _ = data
                timesteps = torch.rand((self.batch_size, 1), device=self.device)
                
                outs = self.model(images.detach(), timesteps)
                loss = torch.mean(outs)    
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.ema.update(self.model)
        
                loss_item = loss.detach().item()
                batch_loss += loss_item
                
                pbar.set_description(f"Training NFDM | Batch Loss: {loss_item:.3f} | Last Loss: {self.epoch_loss:.3f} | LR: {lr:.7f} | Iter: {self.it}")

            self.lr_sched.step()
            self.epoch_loss = batch_loss / len(images)
                
            if checkpoint and batch_loss < max_loss: 
                self.save('checkpoint')
                max_loss = batch_loss

    def save(
        self: Self,
        dir: str = 'nfdm'
    ) -> None: 
        torch.save({
            'nfdm' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'/models/{dir}.pt')
        
    def load(
        self: Self, 
        dir: str = 'nfdm'
    ) -> None: 
        load_obj = torch.load(f'../models/{dir}.pt', weights_only=True)
        self.model.load_state_dict(load_obj['nfdm'])
        self.ema.load_state_dict(load_obj['ema'])
        self.optimizer.load_state_dict(load_obj['optimizer'])

    @torch.no_grad()
    def generate(
        self: Self,
        num_samples: int = 5,
        num_timesteps: int = 300, 
        frame_count: int = 10,
        save_individual_img: bool = False, 
    ) -> None: 
        def display_reverse(images: List, i: int):
            grid = torchvision.utils.make_grid(images, nrow=len(images)//2, normalize=True, value_range=(-1,1))
            grid = grid.permute(1, 2, 0)
            plt.figure(figsize=(24, 18))
            plt.imshow(grid.cpu().numpy())
            plt.axis('off')
            plt.savefig(f'../examples/nfdm_example_{i}.png')
            torch.cuda.empty_cache()
            
        self.model.eval()
        self.ema.module.eval()

        def sde(z_in, t_in):
            x_ = self.predictor(z_in, t_in)

            _, dz, score = self.transform.inverse(z_in, t_in, x_)

            g = self.scheduler(t_in)
            g2 = g ** 2

            drift = score_based_sde_drift(dz, score, g2)

            return drift, g

        z = torch.randn((num_samples, 3, 32, 32), device=self.device)
        x, (_, zs) = solve_sde(sde, z, 1, 0, num_timesteps, show_pbar=True, device=self.device)
        step_size = len(zs) // (frame_count-1)
        for i, image in enumerate(x):
            display_reverse(torch.cat((zs[::step_size,i,:,:,:][:frame_count-1], image.unsqueeze(0))), i)
            
        self.model.train()
        self.ema.module.train()

if __name__ == "__main__": 
    x = torch.randn((1, 3, 32, 32))
    model = NFDM()
    model.model(x, 0)