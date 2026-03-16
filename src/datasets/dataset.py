import torch
from torch.utils.data import Dataset, DataLoader, random_split
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


# DataLoader
def get_dataloaders(csv_path, train, val, test, batch_size, random_seed):
    """
    Instantiates SurvivalDataset and splits it into three DataLoaders
    
    Args:
        csv_path (str): path to the preprocessed data CSV file
        train (float): proportion of data for training set
        val (float): proportion of data for validation set
        test (float): proportion of data for testing set
        batch_size (int): number of samples in each batch
        random_seed (int): seed for random split of dataset
    
    Returns:
        train (shuffled), val, and test DataLoaders
    """
    
    if abs(train + val + test - 1.0) > 1e-6:    # avoid underflow
        raise ValueError("train, val, and test must sum to 1.0")
    
    dataset = SurvivalDataset(csv_path)
    
    # torch split dataset - can take proportions and performs floor multiplication with length
    generator = torch.Generator().manual_seed(random_seed)
    train_data, val_data, test_data = random_split(dataset, [train, val, test], generator)
    
    # create DataLoaders
    train_loader = DataLoader(train_data, batch_size, shuffle=True, 
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, 
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, 
                             num_workers=0, drop_last=False)
    
    return train_loader, val_loader, test_loader