import torch
import numpy as np 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Self, List
from nfdm.model.model_abc import GenerativeMethod
from timm.utils.model_ema import ModelEmaV3
from tqdm import tqdm
from torch.amp import GradScaler
from line_profiler import profile

# implementation loosely inspired by https://medium.com/data-science/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946

class Sinusoidal(nn.Module):
    
    def __init__(
        self: Self,
        embed_size: int, 
        horizon: int = 1000,
        *args, 
        **kwargs
    ) -> None:
        super(Sinusoidal, self).__init__(*args, **kwargs) 
        
        pe = torch.zeros(horizon, embed_size, requires_grad=False)
        positions = torch.arange(0, horizon).unsqueeze(dim=1)
        div = torch.exp(torch.arange(0, embed_size, 2).float() * -(np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        
        self.embed_size = embed_size
        self.register_buffer("pe", pe) 

    def forward(self: Self, t: int) -> Tensor:
        return self.pe[t].view(-1, self.embed_size, 1, 1)
    
    
class ResidualBlock(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int,
        horizon: int = 1000, 
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU,
        groups: int = 32,
        *args, 
        **kwargs
    ) -> None:
        super(ResidualBlock, self).__init__(*args, **kwargs)
        
        self.sinusoidal = Sinusoidal(in_channels, horizon)
        
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self: Self, x: Tensor, t: int, embed: bool) -> Tensor:
        if embed: 
            x = x + self.sinusoidal(t)
        return self.layers(x) + x * 1/np.sqrt(2)
    
    
class UNETLayer(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int, 
        out_channels: int, 
        resblock: ResidualBlock = ResidualBlock,
        upsample: bool = False, 
        attention: bool = False, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        *args, 
        **kwargs
    ) -> None:
        super(UNETLayer, self).__init__(*args, **kwargs)
        
        self.resblock1 = resblock(in_channels=in_channels)    
        self.resblock2 = resblock(in_channels=in_channels)
        
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
        
    def forward(self: Self, x: Tensor, t: int) -> None: 
        x = self.resblock1(x, t, True)
        if self.attention:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2) # attention on patches
            x, _ = self.attention_layer(x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)
    
        x = self.resblock2(x, t, False)
        return self.conv(x), x
    
    
class DDPM(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int = 3, 
        channels: List = list([64, 128, 256, 512, 512, 384, 192]),
        *args, 
        **kwargs
    ) -> None:
        super(DDPM, self).__init__(*args, **kwargs)
        
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
        x = self.conv1(x)
        x, x1 = self.layer1(x, t)
        x, x2 = self.layer2(x, t)
        x, x3 = self.layer3(x, t)

        x, _ = self.layer4(x, t)
        x, _ = self.layer5(torch.cat([x3, x], dim=1), t)
        x, _ = self.layer6(torch.cat([x2, x], dim=1), t)
        x = self.conv2(torch.cat([x1, x], dim=1))
        x = self.relu(x)
        x = self.conv3(x)
        
        return x

# class from the medium article 
class Var_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - beta
        alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        

    def forward(self, t):
        return self.beta[t], self.alpha[t]
    
class Diffusion(GenerativeMethod):
    
    def __init__(
        self: Self, 
        in_channels: int = 3, 
        horizon: int = 1000,
        batch_size: int = 128, 
        lr: float = 2e-4,
        device: str = 'cpu',
        gamma: float = 0.99, 
        amp: bool = True, 
    ) -> None:
        super(Diffusion, self).__init__()
        
        self.model = DDPM(in_channels=in_channels).to(device=device)
        self.ema = ModelEmaV3(self.model, device=device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.variance_sched = Var_Scheduler(horizon).to(device=device)
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.horizon = horizon
        self.amp = amp
        self.lr_sched = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
    
    @profile
    def train(
        self: Self, 
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 100, 
        checkpoint: bool = True, 
    ) -> List[float]: 
        if checkpoint:
            try: 
                self.load('checkpoint_ddpm')
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
                timesteps = torch.randint(0, self.horizon, (self.batch_size, ), device=self.device)
                noise = torch.randn_like(images, device=self.device).detach()
                alpha = self.variance_sched.alpha[timesteps].view(self.batch_size, 1, 1, 1)
                images = (torch.sqrt(alpha) * images) + (torch.sqrt(1-alpha) * noise)
                
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.amp):
                    outs = self.model(images.detach(), timesteps)
                    loss = self.criterion(outs, noise)
                
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                self.ema.update(self.model)
        
                batch_loss += loss.detach().item()


            self.lr_sched.step()
            self.epoch_loss = batch_loss / len(images)
            pbar.set_description(f"Training Diffusion | Last Loss: {self.epoch_loss:.3f} | LR: {lr:.7f}")
                
            if checkpoint and batch_loss < max_loss: 
                self.save('checkpoint_ddpm')
                max_loss = batch_loss
            
            
    def save(
        self: Self,
        dir: str = 'ddpm'
    ) -> None: 
        torch.save({
            'ddpm' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'models/{dir}.pt')
    
    def load(
        self: Self, 
        dir: str = 'ddpm'
    ) -> None: 
        load_obj = torch.load(f'models/{dir}.pt', weights_only=True)
        self.model.load_state_dict(load_obj['ddpm'])
        self.ema.load_state_dict(load_obj['ema'])
        self.optimizer.load_state_dict(load_obj['optimizer'])
    
    # code from medium article 
    @torch.no_grad()
    def generate(
        self: Self,
        num_samples: int = 5,
        num_timesteps: int = 1000, 
        save_individual_img: bool = False, 
    ) -> None: 
        def display_reverse(images: List, i: int):
            grid = torchvision.utils.make_grid(torch.cat(images, dim=0), nrow=len(images)//2, normalize=True, value_range=(-1,1))
            grid = grid.permute(1, 2, 0)
            plt.figure(figsize=(24, 18))
            plt.imshow(grid.cpu().numpy())
            plt.axis('off')
            plt.savefig(f'examples/ddpm_example_{i}.png')
            torch.cuda.empty_cache()
            
        self.model.eval()
        self.ema.module.eval()
        times = [0,15,50,100,200,300,400,550,700,999]
        images = []

        for i in range(num_samples):
            z = torch.randn((1, 3, 32, 32), device=self.device)
            for t in reversed(range(1, num_timesteps)):
                t = [t]
                temp = (self.variance_sched.beta[t]/( (torch.sqrt(1-self.variance_sched.alpha[t]))*(torch.sqrt(1-self.variance_sched.beta[t])) ))
                z = (1/(torch.sqrt(1-self.variance_sched.beta[t])))*z - (temp*self.model(z,t))
                if t[0] in times:
                    images.append(z)
                e = torch.randn((1, 3, 32, 32), device=self.device)
                z = z + (e*torch.sqrt(self.variance_sched.beta[t]))
            temp = self.variance_sched.beta[0]/( (torch.sqrt(1-self.variance_sched.alpha[0]))*(torch.sqrt(1-self.variance_sched.beta[0])) )
            x = (1/(torch.sqrt(1-self.variance_sched.beta[0])))*z - (temp*self.model(z,[0]))

            images.append(x)
            # x = x.squeeze(0).transpose(0, 1).transpose(1, 2).detach()
            # x = x.numpy()
            # # plt.imshow(x)
            # if save_individual_img:
            #     plt.savefig(f'examples/ddpm_image_{i}')
            display_reverse(images, i)
            images = []
            
        self.model.train()
        self.ema.module.train()
        
        
if __name__ == "__main__": 
    x = torch.randn((1, 3, 32, 32))
    model = Diffusion()
    model.model(x, 0)
    
    