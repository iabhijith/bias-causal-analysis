
import torch

from tqdm import tqdm
from data.blimp import BlimpData

def eval_blimp(model, tokenizer, data, device, sample=None, seed=42):
    data = BlimpData(data, sample=sample, seed=seed)
    dataloader = data.get_dataloader()
    model.to(device)
    model.eval()
    acceptable = 0
    count = 0
    with torch.no_grad():
        for a, u in tqdm(dataloader):
            score_x = score(model, tokenizer, a[0], device)
            score_xc = score(model, tokenizer, u[0], device)
            if score_x < score_xc:
                acceptable += 1
            count += 1
    return acceptable / count

def score(model, tokenizer, sentence, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device) 
    outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return sentence_prob