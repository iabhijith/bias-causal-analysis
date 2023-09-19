import torch
import polars as pl
import pandas as pd
import random

from os import path, makedirs
from typing import Tuple

from torch.utils.data import Dataset, DataLoader
from random import shuffle, sample


FILE_NAME = "bug_balanced.csv"
COLUMNS = ['profession', 'g', 'profession_first_index', 'profession_last_index', 'g_first_index', 'g_last_index', 'sentence_text', 'stereotype', 'distance', 'num_of_pronouns', 'predicted gender']

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        return X, y

class BUGBalanced():
    def __init__(self, data_path="./", val_split=0.1, seed=42):
        super()
        file_path = path.join(data_path, FILE_NAME)
        self.train_data, self.val_data = self.prepare_data(file_path=file_path,
                                                           columns=COLUMNS,
                                                           val_split=val_split,
                                                           seed=seed)

    def prepare_data(self, file_path, columns, val_split, seed):
        """
        Prepare data for training and validation
        Parameters
        ----------
        val_split : float
            Percentage of data to be used for validation
        seed : int
            Random seed for reproducibility
        Returns
        -------
        train_data : polars.DataFrame
            Training data
        val_data : polars.DataFrame
            Validation data
        """
        data = pl.scan_csv(source=file_path).select(columns).collect()
        stereo = data.filter(pl.col("stereotype") == 1)
        anti_stereo = data.filter(pl.col("stereotype") == -1)
        stereo_train, stereo_val = self._split(stereo, val_split, seed)
        anti_stereo_train, anti_stereo_val = self._split(anti_stereo, val_split, seed)
        return pl.concat([stereo_train, anti_stereo_train]), pl.concat([stereo_val, anti_stereo_val])

    def _split(self, data: pl.DataFrame, val_split: float, seed: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        data = data.to_pandas()
        val_data = data.sample(frac=val_split, random_state=seed)
        train_data = data.drop(val_data.index)
        return pl.from_pandas(train_data), pl.from_pandas(val_data)
    
    def get_data(self):
        return self.train_data, self.val_data
    
    def get_dataloaders(self, batch_size, shuffle=True):
        train_data = self.train_data.select(["sentence_text", "stereotype"]).rows()
        val_data = self.val_data.select(["sentence_text", "stereotype"]).rows()
        train_dataset = Dataset(train_data)
        val_dataset = Dataset(val_data)
        return (
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), 
        torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        )


if __name__ == '__main__':
    dataset = BUGBalanced()
    train_data, val_data = dataset.get_data()
    train_loader, val_loader = dataset.get_dataloaders(batch_size=10, shuffle=True)
    print(len(train_data))
    print(len(val_data))

   
