import torch 
import numpy as np
import argparse

from nfdm.model.autoencoder import AutoEncoder
from nfdm.utils.dataset import CIFAR10


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=int(1e2), help='number of steps')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-s', '--save', action='store_true', help='save model or not')
parser.add_argument('-d', '--device', type=str, default='cpu', help='device')
parser.add_argument('-t', '--test', action='store_true', help='test mode')
parser.add_argument('--dir', type=str, default=None, help='where to load model from ')

args = parser.parse_args()

if __name__ == "__main__":
    if args.device == 'cuda':
        assert torch.cuda.is_available() == True, "You don't have a CUDA-enabled GPU"
        
    cifar10 = CIFAR10(args.batch_size)
    model = AutoEncoder(in_features=3*32*32, 
                        lr=args.lr, batch_size=args.batch_size, 
                        device=args.device)
    
    idx = np.random.randint(low=0, high=len(cifar10.trainloader))
    test_batch = cifar10.trainloader[idx]
    
    if not args.test:
        model.train(args.epochs, cifar10.trainloader)
        if args.save:
            model.save('ae')
        model.generate(test_batch)
    else:
        assert args.dir is not None, "No model dir provided"
        model.load(args.dir)
        model.generate(test_batch)
        
        
   


