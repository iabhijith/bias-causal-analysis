import polars as pl

from pathlib import Path 
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, Xc = self.data[index]
        return X, Xc


class WinoBiasData():
    def __init__(self, data_path = "winobias",
                 stereo_file="pro_stereotyped_type2.txt.dev",
                 anti_stereo_file="anti_stereotyped_type2.txt.dev"):
        data_path = Path(data_path)
        self.data = self._prepare_data(data_path/stereo_file, data_path/anti_stereo_file)

    def _prepare_data(self, stereo_file, anti_stereo_file):
        stereo =  open(stereo_file, "r")
        anti_stereo = open(anti_stereo_file, "r")
        stereo_lines = [line.rstrip('\n') for line in stereo.readlines()]
        anti_stereo_lines = [line.rstrip('\n') for line in anti_stereo.readlines()] 
        data = list(zip(stereo_lines, anti_stereo_lines))
        return data
    
    def get_dataloader(self):
        dataset = Dataset(self.data)
        return DataLoader(dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":
    data = WinoBiasData()
    dataloader = data.get_dataloader()
    print(len(dataloader)) 
    for X, Xc in dataloader:
        print(X)
        print(Xc)
