import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision

from typing import Self, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
from line_profiler import profile

from nfdm.model.model_abc import GenerativeMethod

class Encoder(nn.Module):
    
    def __init__(
        self: Self,
        in_features: int, 
        bottleneck: int = 64,
        hidden_layers: List = list([1024, 512]),
        *args, 
        **kwargs
    ) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], bottleneck),
        )
        
    def forward(
        self: Self,
        inp: Tensor 
    ) -> Tensor: 
        if len(inp.shape) > 2: 
            inp = inp.flatten(start_dim=1)
        return self.layers(inp)
    
class Decoder(nn.Module):
    
    def __init__(
        self: Self,
        in_features: int, 
        bottleneck: int = 64,
        hidden_layers: List = list([1024, 512]),
        *args, 
        **kwargs
    ) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        
        self.layers = nn.Sequential(
            nn.Linear(bottleneck, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], in_features),
            nn.Tanh()
        )
        
    def forward(
        self: Self,
        input: Tensor 
    ) -> Tensor: 
        return self.layers(input)

class AutoEncoder(GenerativeMethod):
    """Basic AE that flattens the input image and then performs reconstruction
    """
    
    def __init__(
        self, 
        in_features: int, 
        bottleneck: int = 384, 
        lr: float = 1e-3, 
        batch_size: int = 256, 
        device: str = 'cpu',
        amp: bool = True, 
        *args, 
        **kwargs
    ) -> None:
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.device = device
        
        self.encoder = Encoder(in_features=in_features, 
                               bottleneck=bottleneck).to(device)
        
        self.decoder = Decoder(in_features=in_features, 
                               bottleneck=bottleneck).to(device)
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                          lr=lr)
        
        self.batch_size = batch_size
        self.criterion = nn.MSELoss().cuda()
        self.amp = amp
    
    def train(
        self: Self, 
        epochs: int, 
        train_loader: DataLoader,
    ) -> List[float]:
        
        self.encoder.train()
        self.decoder.train()
        
        scaler = GradScaler(self.device, enabled=self.amp)  

        losses = []
        for epoch in (pbar := tqdm(range(epochs))):
            batch_loss = 0
            
            for data in train_loader:
                images, _ = data

                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):  
                    encoded = self.encoder(images)
                    reconstructed = self.decoder(encoded)
                    loss = self.criterion(reconstructed, images.flatten(start_dim=1))
                    
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                self.optimizer.zero_grad()

                # this line takes a lot of runtime cpu-gpu sync
            batch_loss = loss.item() 
            # batch_loss /= self.batch_size 
            # losses.append(batch_loss)
            pbar.set_description(f"Training AE | Last Loss: {batch_loss:.3f}")
            
        # return losses        
    def generate(
        self: Self, 
        batch: Tuple, 
        num_gen: int = 5,
        inp_shape: Tuple = (3, 32, 32)
    ) -> None:
        
        self.encoder.eval()
        self.decoder.eval()
        
        images, _ = batch

        # code from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
        idxs = torch.randint(low=0, high=len(images), size=(num_gen,))
        input_imgs = images[idxs]
        
        encodings = self.encoder(input_imgs.flatten(start_dim=1))
        reconst_imgs = self.decoder(encodings).detach().view(-1, *inp_shape)
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
        grid = torchvision.utils.make_grid(imgs, nrow=num_gen, normalize=True, value_range=(-1,1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(12, 8))
        plt.imshow(grid.cpu().numpy())
        plt.axis('off')
        plt.savefig(f'examples/AE_Reconstructions.png')
        
        
        self.encoder.train()
        self.decoder.train()
    def save(
        self: Self, 
        dir: str,
    ) -> None:
        torch.save({
            'encoder' : self.encoder.state_dict(),
            'decoder' : self.decoder.state_dict()
            }, f'models/{dir}.pt')
        
    def load(
        self: Self, 
        dir: str, 
    ) -> None:
        load_obj = torch.load(f'models/{dir}.pt', weights_only=True)
        self.encoder.load_state_dict(load_obj['encoder'])
        self.decoder.load_state_dict(load_obj['decoder']) 

        
        