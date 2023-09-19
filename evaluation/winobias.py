
import torch

from tqdm import tqdm
from data.winobias import WinoBiasData

def eval_winobias(model, tokenizer, data_path, stereo_file, anti_stereo_file, device):
    data = WinoBiasData(data_path, stereo_file, anti_stereo_file)
    dataloader = data.get_dataloader()
    model.to(device)
    model.eval()
    stereo = 0
    count = 0
    with torch.no_grad():
        for a, u in tqdm(dataloader):
            score_x = score(model, tokenizer, a[0], device)
            score_xc = score(model, tokenizer, u[0], device)
            if score_x > score_xc:
                stereo += 1
            count += 1
    return stereo/ count

def score(model, tokenizer, sentence, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device) 
    outputs = model(input_ids, labels=input_ids)
    return -torch.exp(outputs["loss"]).item()
   