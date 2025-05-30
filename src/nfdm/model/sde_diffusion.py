import torch 
import numpy as np

from typing import Self

from nfdm.model.ddpm import DDPM

class SDEUnet(DDPM):
    
    def __init__(
        self: Self, 
        in_channels: int = 3, 
        *args, 
        **kwargs
    ) -> None:
        super(SDEUnet, self).__init__(in_channels, *args, **kwargs)
        
        