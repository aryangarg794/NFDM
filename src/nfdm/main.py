import torch 
import numpy as np
import argparse
import random

from nfdm.model.autoencoder import AutoEncoder
from nfdm.model.ddpm import Diffusion
from nfdm.model.nfdm_model import NFDM
from nfdm.utils.dataset import CIFAR10


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=int(1e2), help='number of steps')
parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size')
parser.add_argument('-lr', '--lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('-s', '--save', action='store_true', help='save model or not')
parser.add_argument('-d', '--device', type=str, default='cpu', help='device')
parser.add_argument('-t', '--test', action='store_true', help='test mode')
parser.add_argument('-a', '--amp', action='store_false', help='turn off amp mode')
parser.add_argument('--seed', type=int, default=None, help='seed experiment')
parser.add_argument('--dir', type=str, default=None, help='where to load model from ')
parser.add_argument('-m', '--model', type=str, default='nfdm', help='model type ')


args = parser.parse_args()

if __name__ == "__main__":
    if args.device == 'cuda':
        assert torch.cuda.is_available() == True, "You don't have a CUDA-enabled GPU"
    
    print(f'Using {args.device.upper()}')
    
    if args.seed  is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    try: 
        
        match args.model: 
            case 'nfdm':
                model = NFDM(in_channels=3, 
                            lr=args.lr, batch_size=args.batch_size, 
                            device=args.device, amp=args.amp) 
            case 'ae':
                model = AutoEncoder(in_features=3*32*32, 
                            lr=args.lr, batch_size=args.batch_size, 
                            device=args.device, amp=args.amp)
        
            case 'ddpm':
                model = Diffusion(in_channels=3, 
                            lr=args.lr, batch_size=args.batch_size, 
                            device=args.device, amp=False) 
            case _:
                raise NotImplementedError('Model type not found')
            
        
        if not args.test:
            print(f'=============Training {args.model.upper()} with batch {args.batch_size}=============')
            cifar10 = CIFAR10(args.batch_size, device=args.device)
            test_batch = next(iter(cifar10.trainloader))
            model.train(epochs=args.epochs, train_loader=cifar10.trainloader, checkpoint=True)
            if args.save:
                model.save()
            model.generate()
        else:
            assert args.dir is not None, "No model dir provided"
            model.load(args.dir)
            model.generate()
    except KeyboardInterrupt:
        print('Experiment Stopped Prematurely')
        
        
   


