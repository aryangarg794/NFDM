import torch 
import torchvision
import torchvision.transforms as transforms
from line_profiler import profile

from typing import Self

class CIFAR10:
    
    def __init__(
        self: Self,
        batch_size: int = 64,
        test: bool = False,
        num_workers: int = 0,
        in_memory: bool = True, 
        device: str = 'cpu'
    ) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.Normalize((0.5,), (0.5,))
        ])
        

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )

        if in_memory:
            images = []
            labels = []
            for img, label in trainset:
                images.append(img)
                labels.append(label)
            images = torch.stack(images).to(device)
            labels = torch.as_tensor(labels, device=device)
            
            self.trainset = torch.utils.data.TensorDataset(images, labels)
        else:
            self.trainset = trainset

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if device == 'cpu' else False,
            num_workers=num_workers,
            drop_last=True
        )
        
        if test:
            self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            self.testloader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

        
        