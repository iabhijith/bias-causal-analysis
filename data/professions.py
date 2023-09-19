import torch
import random

from torch.utils.data import Dataset, DataLoader
from os import path, makedirs
from sklearn.model_selection import train_test_split

MAN = "man"
WOMAN = "woman"

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, Xc, y = self.data[index]
        return X, Xc, y

class ProfessionsData():
    def __init__(self, data_path="./", seed=42):
        super()
        self.data_path = data_path
        self.data = self.prepare_data()
        self.seed = seed

    def prepare_data(self):
        templates = self.get_template_list()
        male_stereo, female_stereo = self.get_stereo()
        professions = male_stereo + female_stereo
        
        contexts = []
        counter_contexts = []
        labels = []
        for template in templates:
            for profession in professions:
                context = template.format(profession)
                contexts.append(context)
                if profession in male_stereo:
                    counter_contexts.append(template.format(WOMAN))
                    labels.append(1)
                else:
                    counter_contexts.append(template.format(MAN))
                    labels.append(int())
        data = list(zip(contexts, counter_contexts, labels))
        return data
    
    def get_dataloaders(self, batch_size, shuffle=True, val_split=0.1):
        train_data, test_data = train_test_split(self.data, test_size=val_split, random_state=self.seed)
        train_dataset = Dataset(train_data)
        val_dataset = Dataset(test_data)
        return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), 
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        )

    def get_stereo(self):
        male_stereo = []
        female_stereo = []
        with open(path.join(self.data_path, "professions_male_stereo.json"), "r") as f:
            for l in f:
                for j in eval(l):
                    male_stereo.append(j[0])
        with open(path.join(self.data_path, "professions_female_stereo.json"), "r") as f:
            for l in f:
                for j in eval(l):
                    female_stereo.append(j[0])

        return male_stereo, female_stereo


    def get_profession_list(self, indices=None):
        word_list = []
        with open(path.join(self.data_path, "professions.json"), "r") as f:
            for l in f:
                for j in eval(l):
                    word_list.append(j[0])
        return word_list

    def get_template_list(self, indices=None):
        '''
         Get list of all considered templates by Vig et al. 2020
         A subset from https://arxiv.org/pdf/1807.11714.pdf Lu et al. 2020
        ''' 
        templates = [
            "The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because",
        ]
        if indices:
            subset_templates = [templates[i - 1] for i in indices]
            print("subset of templates:", subset_templates)
            return subset_templates
        return templates

if __name__ == '__main__':
    data = ProfessionsData(seed=1)
    train_dataloader, val_dataloader = data.get_dataloaders(batch_size=1, shuffle=True, val_split=0.00001)  
    m, f = data.get_stereo()
    print(len(m))
    print(len(f))
    print(len(train_dataloader.dataset.data))
    print(len(val_dataloader.dataset.data))
    for batch in train_dataloader:
       X, Xc, y = batch
       print(X[0]+ " he")
       print(X[0]+ " she")
       print(y[0].item())
       break