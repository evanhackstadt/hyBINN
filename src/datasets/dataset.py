import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SurvivalDataset(Dataset):
    
    def __init__(self, csv_path: str):
        """
        csv_path: path to the preprocessed data CSV file
        """
        
        df = pd.read_csv(csv_path, index_col=0)
        df_features = df.drop(columns=['OS.time', 'OS'])
        df_labels = df[['OS.time', 'OS']]
        print(df_features.head())
        print(df_labels.head())
        
        # covert dataframe --> numpy
        features = df_features.to_numpy(dtype=np.float32)
        labels = df_labels.to_numpy(dtype=np.float32)
        
        # convert numpy --> tensors
        self.x = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)    # time, event
        self.n_samples = df.shape[0]
        self.feature_dim = self.x.shape[1]
        self.gene_cols = list(df_features.columns)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {
            'x': self.x[idx],
            'time': self.y[idx, 0],
            'event': self.y[idx, 1]
        }
        return item