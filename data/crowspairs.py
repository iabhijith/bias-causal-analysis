import pandas as pd
import torch
from torch.utils.data import Dataset


class CrowSPairsDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()    
        df = pd.read_csv(data_path, sep='\t')

        # if direction is stereo, sent1, sent2 are sent_more, sent_less respectively,
        # otherwise the other way around
        df["direction"] = df["stereo_antistereo"]
        df["sent1"] = df["sent_less"]
        df["sent2"] = df["sent_more"]
        df.loc[df["direction"] == "stereo", "sent1"] = df["sent_more"]
        df.loc[df["direction"] == "stereo", "sent2"] = df["sent_less"]

        # Convert dataframe to list of dictionaries
        self.items = df[["sent1", "sent2", "direction", "bias_type"]].to_dict("records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
    

    if __name__ == "__main__":
        df = pd.read_csv("crows_pairs_revised.csv", sep='\t')
        print(df.shape)
        df_g = df[df["bias_type"] == "gender" ]
        print(df_g.shape)

