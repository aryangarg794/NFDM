import torch 
import math 
import numpy as np
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Self, List, Tuple, Callable
from timm.utils.model_ema import ModelEmaV3
from tqdm import tqdm
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torchsummary import summary

from nfdm.model.model_abc import GenerativeMethod
from nfdm.model.ddpm import UNETLayer


class WarmupPolyDecayLR(LRScheduler):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float,
        warmup_target_lr: float,
        decay_epochs: int,
        final_lr: float,
        last_epoch: int = -1
    ):
        self.warmup_epochs   = warmup_epochs
        self.decay_epochs    = decay_epochs
        self.start_lr        = warmup_start_lr
        self.target_lr       = warmup_target_lr
        self.final_lr        = final_lr

        for group in optimizer.param_groups:
            group["lr"] = self.start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch  

        if epoch < 0:
            return [self.start_lr for _ in self.optimizer.param_groups]
        
        if epoch <= self.warmup_epochs:
            lr = self.target_lr + (self.start_lr - self.target_lr) * max(0, (self.warmup_epochs - epoch) / self.warmup_epochs)

        elif self.warmup_epochs <= epoch <= self.warmup_epochs + self.decay_epochs:
            decay_step = epoch - self.warmup_epochs
            lr = self.final_lr + (self.target_lr - self.final_lr) * max(0, (self.decay_epochs - decay_step) / self.decay_epochs)
        else:
            lr = self.final_lr

        return [lr for _ in self.optimizer.param_groups]
        

class VolatilityLinSNR(nn.Module):
    def forward(self, t: Tensor) -> Tensor:
        return (20 * torch.sigmoid(-10 + 20 * t)) ** 0.5
        

class VolatilityNeural(nn.Module):
    def __init__(self: Self, hidden_dim: int = 64, in_dim: int =1, out_dim: int = 1  ):
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
        return self.sp(self.net(t))

class FourierTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.freqs = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self: Self, t: float | Tensor):
        angles = t * self.freqs * 2 * np.pi  
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb  
    
class ResidualBlockSDE(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int,
        dropout: float = 0.0, 
        activation: nn.Module = nn.ReLU,
        groups: int = 32,
        hidden_dim: int = 64, 
        *args, 
        **kwargs
    ) -> None:
        super(ResidualBlockSDE, self).__init__(*args, **kwargs)
        
        self.time = nn.Sequential(
            FourierTimestepEmbedding(in_channels), 
        )
        
        self.embed_size = in_channels
        
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self: Self, x: Tensor, t: float | Tensor, embed: bool) -> Tensor:
        if embed: 
            x = x + self.time(t).view(-1, self.embed_size, 1, 1)
        return self.layers(x) + x * 1/np.sqrt(2)
    
class UNETNFDM(UNETLayer):
    
    def __init__(self, dropout: float = 0.0, *args, **kwargs):
        super(UNETNFDM, self).__init__(resblock = ResidualBlockSDE,  dropout=dropout, *args, **kwargs)

class ForwardNet(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int = 3, 
        delta: float = 1e-2, 
        channels: List = list([64, 128, 256, 512, 512, 384, 192]),
        *args, 
        **kwargs
    ) -> None:
        super(ForwardNet, self).__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = UNETNFDM(in_channels=64, out_channels=128) # 64 -> 128 
        self.layer2 = UNETNFDM(in_channels=128, out_channels=256, attention=True) # 128 -> 256
        self.layer3 = UNETNFDM(in_channels=256, out_channels=512) # 256 -> 512
        self.layer4 = UNETNFDM(in_channels=512, out_channels=256, upsample=True) # 512 -> 256
        self.layer5 = UNETNFDM(in_channels=512, out_channels=256, upsample=True) # 512 -> 384
        self.layer6 = UNETNFDM(in_channels=384, out_channels=128, attention=True, upsample=True) # 384 -> 192 
        
        self.conv2 = nn.Conv2d(in_channels=channels[6], out_channels=channels[6]//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels[6]//2, out_channels=2 * in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
        self.delta = delta
        self.in_channels = in_channels
        self.sp = nn.Softplus()

        
    def forward(self: Self, x: Tensor, t: float | Tensor) -> Tensor:
        h = self.conv1(x)
        h, x1 = self.layer1(h, t)
        h, x2 = self.layer2(h, t)
        h, x3 = self.layer3(h, t)

        h, _ = self.layer4(h, t)
        h, _ = self.layer5(torch.cat([x3, h], dim=1), t)
        h, _ = self.layer6(torch.cat([x2, h], dim=1), t)
        h = self.conv2(torch.cat([x1, h], dim=1))
        h = self.relu(h)
        h = self.conv3(h)
        
        mu_bar, sigma_bar = h.chunk(2, dim=1)
        sigma_bar = self.sp(sigma_bar)
        time_tensor = t.view(-1, 1, 1, 1)
        mu = (1 - time_tensor) * x + time_tensor * (1 - time_tensor) * mu_bar
        sig = math.log(self.delta) * (1 - time_tensor) + sigma_bar * time_tensor * (1 - time_tensor) # avoiding nans 
        # sig = sig.clamp(-5, 5)
        
        return mu, self.sp(sig)

class ForwardProcess(nn.Module):
    
    def __init__(
        self: Self, 
        in_channels: int = 3, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.net = ForwardNet(in_channels)
        
    def jvp(self: Self, f: Callable, x: Tensor, v: Tensor) -> Tuple[Tensor]:
        return torch.autograd.functional.jvp(
            f, x, v, 
            create_graph=torch.is_grad_enabled()
        )

    def t_dir(self: Self, f: Callable, t: Tensor) -> Tuple[Tensor]:
        return self.jvp(f, t, torch.ones_like(t))    

    def get_jvp(self: Self, x: Tensor, t: float | Tensor) -> Tuple:
        def func(x_in): 
           def func_t(t_in):
               return self.net(x_in, t_in)
           return func_t

        return self.t_dir(func(x), t)
    
    def forward(
        self: Self, 
        eps: Tensor, 
        x: Tensor,
        t: float | Tensor, 
    ) -> Tuple[Tensor, Tensor, Tensor]:
        values, grads = self.get_jvp(x, t)

        mu, sigma = values
        dmu, dsigma = grads
        
        # print(sigma.min())  
        z = eps * sigma + mu
        dz = dmu + dsigma * eps
        score = - eps / sigma
        
        return z, dz, score
        
    def inverse(
        self: Self, 
        z: Tensor, 
        x: Tensor,
        t: float | Tensor, 
    ) -> Tuple[Tensor, Tensor, Tensor]:
        values, grads = self.get_jvp(x, t)
        mu, sigma = values
        dmu, dsigma = grads
        
        # print(sigma.min())
        eps = (z - mu) / sigma
        dz = dmu + dsigma / sigma * (z - mu)
        score = (mu - z) / sigma ** 2
        
        return eps, dz, score  

class ReverseProcess(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int = 3, 
        channels: List = list([64, 128, 256, 512, 512, 384, 192]),
        *args, 
        **kwargs
    ) -> None:
        super(ReverseProcess, self).__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = UNETNFDM(in_channels=64, out_channels=128) # 64 -> 128 
        self.layer2 = UNETNFDM(in_channels=128, out_channels=256, attention=True) # 128 -> 256
        self.layer3 = UNETNFDM(in_channels=256, out_channels=512) # 256 -> 512
        self.layer4 = UNETNFDM(in_channels=512, out_channels=256, upsample=True) # 512 -> 256
        self.layer5 = UNETNFDM(in_channels=512, out_channels=256, upsample=True) # 512 -> 384
        self.layer6 = UNETNFDM(in_channels=384, out_channels=128, attention=True, upsample=True)  # 384 -> 192 
        
        self.conv2 = nn.Conv2d(in_channels=channels[6], out_channels=channels[6]//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels[6]//2, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self: Self, x: Tensor, t: int) -> Tensor:
        h = self.conv1(x)
        h, x1 = self.layer1(h, t)
        h, x2 = self.layer2(h, t)
        h, x3 = self.layer3(h, t)

        h, _ = self.layer4(h, t)
        h, _ = self.layer5(torch.cat([x3, h], dim=1), t)
        h, _ = self.layer6(torch.cat([x2, h], dim=1), t)
        h = self.conv2(torch.cat([x1, h], dim=1))
        h = self.relu(h)
        out = self.conv3(h)
        time = t.view(-1, 1, 1, 1)
        # out = (1 - time) * out + (time + 0.01) * x
        
        return out
    
    
class NFDMModel(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int = 3,
    ) -> None:
        super(NFDMModel, self).__init__()
        
        self.forward_process = ForwardProcess(in_channels)
        self.reverse_process = ReverseProcess(in_channels)
        self.g_t = VolatilityNeural()
        
    def drift(self: Self, dz: Tensor, score: Tensor, vol: Tensor) -> Tensor:
        return dz + 0.5 * vol * score
    
    def forward(self: Self, x: Tensor, t: Tensor) -> Tensor: 
        eps = torch.randn_like(x)
        
        z, forward_dz, forward_scores = self.forward_process(eps, x, t)
        pred_x = self.reverse_process(z, t)
        _, reverse_dz, reverse_scores = self.forward_process.inverse(z, pred_x, t)
        
        vol = self.g_t(t).pow(2).view(-1, 1, 1, 1)
        # with torch.no_grad():
        #     print("g_t(t) min/mean/max:", self.g_t(t).min().item(), self.g_t(t).mean().item(), self.g_t(t).max().item())
        #     print("vol      min/mean/max:", vol.min().item(),   vol.mean().item(),   vol.max().item())
        # print(vol.min())
        forward_drift  = self.drift(forward_dz, forward_scores, vol)
        reverse_drift  = self.drift(reverse_dz, reverse_scores, -vol)
        
        losses = 0.5 * (forward_drift - reverse_drift).pow(2) / vol
        
        return losses
    
class NFDM(GenerativeMethod):
    
    def __init__(
        self: Self,
        in_channels: int = 3,
        warmup: int = 10,
        batch_size: int = 128, 
        lr: float = 2e-4,
        warmup_lr: float = 1e-8,
        device: str = 'cpu',
        amp: bool = True, 
    ) -> None:
        super().__init__()
        
        self.model = NFDMModel(in_channels).to(device)
        self.ema = ModelEmaV3(self.model, device=device)
    
        self.batch_size = batch_size
        self.device = device
        self.amp = amp
        self.warmup = warmup
        self.lr = lr
        self.warmup_lr = warmup_lr
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    
    def train(
        self: Self, 
        epochs: int, 
        train_loader: torch.utils.data.DataLoader,
        checkpoint: bool = True,
    ) -> None:
        if checkpoint:
            try: 
                self.load('checkpoint_nfdm')
            except FileNotFoundError:
                print('No checkpoint found')
        
        max_loss = float('inf')
        self.it = 0 
        self.epoch_loss = 0   
        # self.lr_sched = WarmupPolyDecayLR(self.optimizer, self.warmup, self.warmup_lr, self.lr, 
        #                                   epochs-1, self.warmup_lr)
        
        self.lr_sched = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.warmup + epochs)
        loss_array = []
        images_high_loss = None
        images_nan = None
        
        for epoch in (pbar := tqdm(range(epochs + self.warmup))): 
            batch_loss = 0 
            lr = self.lr_sched.get_last_lr()[0]
            
            for data in train_loader:
                self.it += 1
                images, _ = data
                
                timesteps = torch.rand((self.batch_size, 1), device=self.device)
                losses = self.model(images, timesteps)
                loss = losses.mean()
                    
                
                self.optimizer.zero_grad()
                loss.backward()
                # for name, param in self.model.named_parameters():
                #     print(f'Iteration: {self.it} | Epoch: {epoch} | Name: {name} | Param Norm: {param.norm()}')
                
                self.optimizer.step()
                
                # self.ema.update(self.model)
                loss_item = loss.detach().item()
                loss_array.append(loss_item)
                # print(f'Loss for Iteration {self.it} and Epoch {epoch}: {loss_item}')
                # if loss > 1e8: 
                #     if images_high_loss: 
                #         images_high_loss = torch.cat([images_high_loss, images], dim=0)
                #     else:
                #         images_high_loss = images
                
                # if torch.isnan(loss).any():
                #     images_nan = images
                #     break
                # batch_loss += loss.detach().item()
                
                pbar.set_description(f"Training NFDM | Last Loss: {loss:.3f} | LR: {lr:.9f} | Iter: {self.it}")

            if torch.isnan(loss).any():
                break

            
            self.epoch_loss = batch_loss / len(images)
            # self.lr_sched.step()    
            if checkpoint and batch_loss < max_loss: 
                self.save('checkpoint_nfdm')
                max_loss = batch_loss
        
        # nan_indices = np.where(np.isnan(loss_array))[0]

        # if nan_indices.size > 0:
        #     first_nan = nan_indices[0]
        #     plt.plot(np.arange(first_nan), loss_array[:first_nan], label="Training Curve")
        #     plt.axvline(x=first_nan, color='r', linestyle='--', label="First NaN")
        # else:
        #     plt.plot(loss_array, label="No Nans training curve")

        # plt.xlabel("Iterations")
        # plt.ylabel("Loss")
        # plt.title("Training Curve Plot with NaN Marker")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('losses.png')
        
        # self.plot_batch(images_nan, 'nan_batch')
        # self.plot_batch(images_high_loss, 'high_loss_batch')

    def save(
        self: Self,
        dir: str = 'nfdm'
    ) -> None: 
        torch.save({
            'nfdm' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'models/{dir}.pt')
        
    def plot_batch(self: Self, images: Tensor, name: str) -> None:
        grid = torchvision.utils.make_grid(images, nrow=len(images)//2, normalize=True, value_range=(-1,1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(24, 18))
        plt.imshow(grid.cpu().numpy())
        plt.axis('off')
        plt.savefig(f'{name}.png')
    
    def load(
        self: Self, 
        dir: str = 'nfdm'
    ) -> None: 
        load_obj = torch.load(f'models/{dir}.pt', weights_only=True)
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
            plt.savefig(f'examples/nfdm_example_{i}.png')
            torch.cuda.empty_cache()
            
        self.model.eval()
        self.ema.module.eval()

        def sde(z_in, t_in):
            x_ = self.model.reverse_process(z_in, t_in)

            _, dz, score = self.model.forward_process.inverse(z_in, x_, t_in)

            g = self.model.g_t(t_in)
            g2 = g ** 2

            drift = self.model.drift(dz, score, g2)

            return drift, g

        z = torch.randn((num_samples, 3, 32, 32), device=self.device)
        x, (_, zs) = self.solve_sde(sde, z, 1, 0, num_timesteps, show_pbar=True, device=self.device)
        step_size = len(zs) // (frame_count-1)
        for i, image in enumerate(x):
            display_reverse(torch.cat((zs[::step_size,i,:,:,:][:frame_count-1], image.unsqueeze(0))), i)
            
        self.model.train()
        self.ema.module.train()

    @torch.no_grad()
    def solve_sde(
            self: Self,     
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
    
if __name__ == "__main__": 
    x = torch.randn((1, 3, 32, 32), device='cuda')
    t = torch.rand((1, 1), device='cuda')
    model = NFDMModel().to('cuda')
    
    
    out = model(x, t)
    print(summary(model))
    # print(out[0], out[1])    
