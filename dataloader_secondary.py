from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import string
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
class MyCustomDataset(Dataset):
    def __init__(self, X, Y):
        temp_array = np.zeros((X.shape[0],X.shape[1] + 1))
        temp_array[:,:-1] = X
        temp_array[:,-1] = Y
        self.data = temp_array
        self.data = shuffle(self.data)

        
    def __getitem__(self, index):
        X = self.data[index,:-1]
        Y = self.data[index, -1]


        X = X.astype(np.float32)
        X = torch.from_numpy(X)
        Y = torch.tensor(Y).long()
        return X, Y

    def __len__(self):
        return self.data.shape[0]

    