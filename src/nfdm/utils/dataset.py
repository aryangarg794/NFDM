import torch 
import torchvision
import torchvision.transforms as transforms

from typing import Self

class CIFAR10:
    
    def __init__(
        self: Self,
        batch_size: int = 64,
    ) -> None:

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]) # from torch website
        
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True)
        
        