import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

class SurvivalDataset(Dataset):
    
    def __init__(self, x_mapped, x_unmapped, times, events):
        """
        Args:
            x_mapped:   numpy array (n_patients, n_mapped_genes)
            x_unmapped: numpy array (n_patients, n_unmapped_genes)
            times:      numpy array (n_patients,)
            events:     numpy array (n_patients,)
        """
        
        # convert numpy --> tensors
        self.x_mapped   = torch.tensor(x_mapped, dtype=torch.float32)
        self.x_unmapped = torch.tensor(x_unmapped, dtype=torch.float32)
        self.y_time     = torch.tensor(times, dtype=torch.float32)
        self.y_event    = torch.tensor(events, dtype=torch.float32)
        
        self.n_samples = len(events)
        self.mapped_dim = self.x_mapped.shape[1]
        self.unmapped_dim = self.x_unmapped.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {
            'X_mapped': self.x_mapped[idx],
            'X_unmapped': self.x_unmapped[idx],
            'y_time': self.y_time[idx],
            'y_event': self.y_event[idx]
        }
        return item     # returns a dict


# DataLoader
def get_dataloaders(x_mapped, x_unmapped, times, events,
                    train, val, test, batch_size, random_seed):
    """
    Instantiates SurvivalDataset and splits it into three DataLoaders
    
    Args:
        x_mapped (np array): mapped gene data 2D array
        x_unmapped (np array): unmapped gene data 2D array
        times (np array): survival times array
        events (np array): survival events array (1=died, 0=censored)
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
    
    dataset = SurvivalDataset(x_mapped, x_unmapped, times, events)
    
    # torch split dataset - can take proportions and performs floor multiplication with length
    generator = torch.Generator().manual_seed(random_seed)
    train_data, val_data, test_data = random_split(dataset, [train, val, test], generator)
    
    # create DataLoaders
    train_loader = DataLoader(train_data, batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, 
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, 
                             num_workers=0, drop_last=False)
    
    return train_loader, val_loader, test_loader