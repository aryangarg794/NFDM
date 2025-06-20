import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SquarePointDataset(Dataset):
    def __init__(self, data: np.ndarray):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ToyDataCreator:
    def __init__(
        self,
        bottom_left_list=[(20, 0)],
        side_length=10,
        points_per_square=10000,
        seed=42,
        batch_size=128,
        shuffle=True,
        datapath=None
    ):
        self.bottom_left_list = bottom_left_list
        self.side_length = side_length
        self.points_per_square = points_per_square
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datapath = datapath
        if self.datapath is not None:
            self.dataset = SquarePointDataset(torch.load(self.datapath))
        else:
            print("Generating dataset with the following parameters:")
            self.dataset = self._create_dataset()
            torch.save(self.dataset.data, "squares_single_dataset.pt")

        # Create DataLoader for batching
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def plot(self, plot_all=False):   
        """
        Plot a sample batch of points from the dataset.
        """
        sample_batch = next(iter(self.loader))
        if plot_all:
            sample_batch = self.dataset.data
        plt.figure(figsize=(8, 8))
        plt.scatter(sample_batch[:, 0].numpy(), sample_batch[:, 1].numpy(), s=1)
        plt.title('Sample Points from Toy Dataset')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axis('equal')
        plt.grid()
        plt.show()

    def _generate_squares(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        all_points = []

        for (x0, y0) in self.bottom_left_list:
            x = np.random.uniform(x0, x0 + self.side_length, self.points_per_square)
            y = np.random.uniform(y0, y0 + self.side_length, self.points_per_square)
            points = np.stack([x, y], axis=1)
            print(len(points), "points generated for square at bottom left:", (x0, y0))
            all_points.append(points)

        return np.vstack(all_points)

    def _create_dataset(self):
        data = self._generate_squares()
        return SquarePointDataset(data)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
