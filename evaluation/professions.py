
import torch

from tqdm import tqdm
from data.professions import ProfessionsData

def eval_professions(model, tokenizer, data_path, device):
    data = ProfessionsData(data_path)
    test_dataloader, val_dataloader  = data.get_dataloaders(batch_size=1, shuffle=False, val_split=0.00001)
    model.to(device)
    model.eval()
    stereo = 0
    count = 0
    with torch.no_grad():
        for X, Xc, y in tqdm(test_dataloader):
            if y[0].item() == 1:
                a = X[0]+ " he"
                u = X[0]+ " she"
            else:
                a = X[0]+ " she"
                u = X[0]+ " he"
            score_x = score(model, tokenizer, a, device)
            score_xc = score(model, tokenizer, u, device)
            if score_x > score_xc:
                stereo += 1
            count += 1
    return stereo/ count

def score(model, tokenizer, sentence, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device) 
    outputs = model(input_ids, labels=input_ids)
    return -torch.exp(outputs["loss"]).item()