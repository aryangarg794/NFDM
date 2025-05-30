import torch 
import math 
import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Self, List, Tuple
from timm.utils.model_ema import ModelEmaV3
from tqdm import tqdm
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from nfdm.model.model_abc import GenerativeMethod


class WarmUp(LRScheduler):
    
    def __init__(
        self: Self, 
        optimizer: torch.optim, 
        last_epoch: int = -1
    ):
        super().__init__(optimizer, last_epoch)
        
        

class VolatilityNeural(nn.Module):
    def __init__(self: Self, hidden_dim: int = 64, activation: nn.Module = nn.SELU):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, 1)
        )
        self.sp = nn.Softplus()

    def forward(self, t: Tensor) -> Tensor:
        return self.sp(self.net(t))

class FourierTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.freqs = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self: Self, t: float | Tensor):
        angles = t * self.freqs * 2 * np.pi  
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb  
    
class ResidualBlock(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int,
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU,
        groups: int = 32,
        hidden_dim: int = 64, 
        *args, 
        **kwargs
    ) -> None:
        super(ResidualBlock, self).__init__(*args, **kwargs)
        
        self.time = nn.Sequential(
            FourierTimestepEmbedding(in_channels), 
            activation(), 
            nn.Linear(in_channels, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, hidden_dim), 
            activation(),
            nn.Linear(hidden_dim, in_channels)
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
    
    
class UNETLayer(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int, 
        out_channels: int, 
        upsample: bool = False, 
        attention: bool = False, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        *args, 
        **kwargs
    ) -> None:
        super(UNETLayer, self).__init__(*args, **kwargs)
        
        self.resblock1 = ResidualBlock(in_channels=in_channels)    
        self.resblock2 = ResidualBlock(in_channels=in_channels)
        
        if upsample:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=2, padding=1)
            
        self.attention = attention
        if attention: 
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=in_channels, 
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
    def forward(self: Self, x: Tensor, t: float | Tensor) -> None: 
        x = self.resblock1(x, t, True)
        if self.attention:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2) # attention on patches
            x, _ = self.attention_layer(x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)
    
        x = self.resblock2(x, t, False)
        return self.conv(x), x
    
    
class ForwardNet(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int = 3, 
        delta: float = 1e-3, 
        channels: List = list([64, 128, 256, 512, 512, 384, 192]),
        *args, 
        **kwargs
    ) -> None:
        super(ForwardNet, self).__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = UNETLayer(in_channels=64, out_channels=128) # 64 -> 128 
        self.layer2 = UNETLayer(in_channels=128, out_channels=256, attention=True) # 128 -> 256
        self.layer3 = UNETLayer(in_channels=256, out_channels=512) # 256 -> 512
        self.layer4 = UNETLayer(in_channels=512, out_channels=256, upsample=True) # 512 -> 256
        self.layer5 = UNETLayer(in_channels=512, out_channels=256, upsample=True) # 512 -> 384
        self.layer6 = UNETLayer(in_channels=384, out_channels=128, attention=True, upsample=True) # 384 -> 192 
        
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
        time_tensor = t.view(-1, 1, 1, 1)
        mu = (1 - time_tensor) * x + time_tensor * (1 - time_tensor) * mu_bar
        sig = math.log(self.delta) * (1 - time_tensor) + sigma_bar * time_tensor * (1 - time_tensor) # avoiding nans 
        
        return mu, sig.exp()

class ForwardProcess(nn.Module):
    
    def __init__(
        self: Self, 
        in_channels: int = 3, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.net = ForwardNet(in_channels)
        
    def get_jvp(self: Self, x: Tensor, t: float | Tensor) -> Tuple:
        def func(x_in): 
           def func_t(t_in):
               return self.net(x_in, t_in)
           return func_t

        return torch.autograd.functional.jvp(func(x), (t,), (torch.ones_like(t),), create_graph=True) # get dm/dt and ds/dt
        
        
    def forward(
        self: Self, 
        eps: Tensor, 
        x: Tensor,
        t: float | Tensor, 
    ) -> Tuple[Tensor, Tensor, Tensor]:
        values, grads = self.get_jvp(x, t)

        mu, sigma = values
        dmu, dsigma = grads
        
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
        self.layer1 = UNETLayer(in_channels=64, out_channels=128) # 64 -> 128 
        self.layer2 = UNETLayer(in_channels=128, out_channels=256, attention=True) # 128 -> 256
        self.layer3 = UNETLayer(in_channels=256, out_channels=512) # 256 -> 512
        self.layer4 = UNETLayer(in_channels=512, out_channels=256, upsample=True) # 512 -> 256
        self.layer5 = UNETLayer(in_channels=512, out_channels=256, upsample=True) # 512 -> 384
        self.layer6 = UNETLayer(in_channels=384, out_channels=128, attention=True, upsample=True) # 384 -> 192 
        
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
        
    def drift(self: Self, dz: Tensor, score: Tensor, vol: Tensor):
        return dz - 0.5 * vol.view(-1, 1, 1, 1) * score
    
    def forward(self: Self, x: Tensor, t: Tensor) -> Tensor: 
        eps = torch.randn_like(x)
        
        z, forward_dz, forward_scores = self.forward_process(eps, x, t)
        pred_x = self.reverse_process(z, t)
        _, reverse_dz, reverse_scores = self.forward_process.inverse(z, pred_x, t)
        
        vol = self.g_t(t).pow(2)
        forward_drift = self.drift(forward_dz, forward_scores, vol)
        reverse_drift = self.drift(reverse_dz, reverse_scores, vol)
        
        return forward_drift, reverse_drift, vol 
    
class NFDM(GenerativeMethod):
    
    def __init__(
        self: Self,
        in_channels: int = 3,
        warmup: int = 10,
        batch_size: int = 128, 
        lr: float = 2e-4,
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
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_sched = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    
    def train(
        self: Self, 
        epochs: int, 
        train_loader: torch.utils.data.DataLoader,
        checkpoint: bool = False,
    ) -> None:
        if checkpoint:
            try: 
                self.load('checkpoint_nfdm')
            except FileNotFoundError:
                print('No checkpoint found')
        
        max_loss = float('inf')
        self.it = 0 
        self.epoch_loss = 0   
        scaler = GradScaler(self.device, enabled=self.amp)
        
        for epoch in (pbar := tqdm(range(epochs + self.warmup))): 
            batch_loss = 0 
            lr = self.lr_sched.get_last_lr()[0]
            
            for data in train_loader:
                self.it += 1
                images, _ = data
                
                timesteps = torch.rand((self.batch_size, 1), device=self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.amp):
                    forward_drift, reverse_drift, vol = self.model(images, timesteps)
                    loss = (0.5 * (forward_drift - reverse_drift) ** 2 / vol.view(-1, 1, 1, 1)).sum(dim=(1, 2, 3)).mean()
                
                print(loss)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.ema.update(self.model)
        
                batch_loss += loss.detach().item()
                
                pbar.set_description(f"Training NFDM | Last Loss: {self.epoch_loss:.3f} | LR: {lr:.7f} | Iter: {self.it}")

            self.lr_sched.step()
            self.epoch_loss = batch_loss / len(images)
                
            if checkpoint and batch_loss < max_loss: 
                self.save('checkpoint_nfdm')
                max_loss = batch_loss
        
    def save(
        self: Self,
        dir: str = 'nfdm'
    ) -> None: 
        torch.save({
            'nfdm' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'models/{dir}.pt')
    
    def load(
        self: Self, 
        dir: str = 'nfdm'
    ) -> None: 
        load_obj = torch.load(f'models/{dir}.pt', weights_only=True)
        self.model.load_state_dict(load_obj['nfdm'])
        self.ema.load_state_dict(load_obj['ema'])
        self.optimizer.load_state_dict(load_obj['optimizer'])
        
    def generate(self: Self, num_samples: int = 5) -> None: 
        pass

    
if __name__ == "__main__": 
    x = torch.randn((1, 3, 32, 32), device='cuda')
    t = torch.rand((1, 1), device='cuda')
    model = NFDMModel().to('cuda')
    
    out = model(x, t)
    # print(out[0], out[1])    
