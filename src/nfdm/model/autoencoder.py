import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import Self, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

class Encoder(nn.Module):
    
    def __init__(
        self: Self,
        in_features: int, 
        bottleneck: int = 64,
        hidden_layers: List = list([512, 512]),
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
        hidden_layers: List = list([512, 512]),
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
        )
        
    def forward(
        self: Self,
        input: Tensor 
    ) -> Tensor: 
        return self.layers(input)

class AutoEncoder:
    """Basic AE that flattens the input image and then performs reconstruction
    """
    
    def __init__(
        self, 
        in_features: int, 
        bottleneck: int = 64, 
        lr: float = 1e-3, 
        batch_size: int = 256, 
        device: str = 'cpu',
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
        self.criterion = nn.MSELoss()
        self.encoded_values = []
        
    def train(
        self: Self, 
        epochs: int, 
        train_loader: DataLoader,
    ) -> List[float]:
        
        for epoch in (pbar := tqdm(range(epochs))):
            batch_loss = 0
            for i, data in enumerate(train_loader):
                images, _ = data

                encoded = self.encoder(images.to(self.device))
                reconstructed = self.decoder(encoded)
                
                if epoch == epochs-2: # last iteration
                    self.encoded_values.append((encoded[0], images[0])) # add the first of the batch
                
                loss = self.criterion(reconstructed, images.flatten(start_dim=1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_loss += loss.item()
                
            batch_loss /= self.batch_size
            pbar.set_description(f"Training AE | Loss {batch_loss:.3f}")
    def generate(
        self: Self, 
        batch: Tuple, 
        num_gen: int = 10,
        inp_shape: Tuple = (3, 32, 32)
    ) -> None:
        
        self.encoder.eval()
        self.decoder.eval()
        
        images, _ = batch
        
        for i in range(num_gen):
            idx = np.random.randint(low=0, high=len(images))
            encoded_value = self.encoder(images[idx])
            out = self.decoder(encoded_value)
            reconstructed_image = out.view(*inp_shape).numpy()
            true_image = self.encoded_values[idx][1].view(*inp_shape).numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(reconstructed_image)
            axes[0].title('Reconstructed ')
            axes[0].axis('off')
            
            axes[1].imshow(true_image)
            axes[1].title('True Image')
            axes[1].axis('off')
            
            fig.savefig(f'examples/random_example_{i}.png')
            plt.close(fig)

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

        
        