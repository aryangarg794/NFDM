from abc import ABC, abstractmethod
from typing import Self

class GenerativeMethod(ABC):
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def train(self: Self, epochs: int, train_loader) -> None:
        pass
    
    @abstractmethod
    def generate(self: Self) -> None:
        pass
    
    @abstractmethod
    def save(self: Self) -> None:
        pass
    
    @abstractmethod
    def load(self: Self) -> None:
        pass