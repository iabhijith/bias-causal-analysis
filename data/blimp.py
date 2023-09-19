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

class BlimpData():
     def __init__(self, data_path="./blimp", sample=None, seed=42):
        self.sample = sample
        self.seed = seed
        self.data = self._prepare_data(data_path, sample, seed)

     def _prepare_data(self, data_path, sample, seed):
        data_path = Path(data_path)
        queries = []
        if data_path.is_dir(): 
            list_of_files = list(data_path.glob('*.jsonl'))   
            for f in list_of_files:
                q = pl.scan_ndjson(f).select(["sentence_good", "sentence_bad"])
                queries.append(q)
        elif data_path.is_file():
            q = pl.scan_ndjson(data_path).select(["sentence_good", "sentence_bad"])
            queries.append(q)

        data = pl.collect_all(queries)
        data = pl.concat(data).to_pandas()
        if sample is not None:
            data = data.sample(n=self.sample, random_state=seed)
        return data.values.tolist()
     
     def get_dataloader(self):
            dataset = Dataset(self.data)
            return DataLoader(dataset, batch_size=1, shuffle=False)
           

if __name__ == '__main__':
    data = BlimpData("./blimp/anaphor_gender_agreement.jsonl", sample=10)
    dataloader = data.get_dataloader()
    for X, Xc in dataloader:
        print(X)
        print(Xc)
      

       
        
    