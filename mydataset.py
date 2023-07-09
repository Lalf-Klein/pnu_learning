import torch
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
    
    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, idx):
        return np.expand_dims(self.train_x[idx], 0), self.train_y[idx]
    