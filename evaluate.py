import torch
import logging
import json
import yaml

import numpy as np

from pathlib import Path 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from discovery.cma import CMA
from configuration.cma import CMAConfig, DataConfig
from data.professions import ProfessionsData

from evaluation.stereoset import eval_stereoset
from evaluation.blimp import eval_blimp
from evaluation.cpairs import evaluate_crowspairs
from evaluation.winobias import eval_winobias
from evaluation.professions import eval_professions

LOGGER = logging.getLogger(__name__)



def evaluate(model,
             tokenizer,
             device,
             bias_type,
             stereo_data,
             blimp_data,
             crowspairs_data):
    results = {}
    ss_results = eval_stereoset(model=model, tokenizer=tokenizer, data=stereo_data, bias=bias_type, device=device)
    results['LM Score'] = ss_results[bias_type]['LM Score']
    results['SS Score'] = ss_results[bias_type]['SS Score']
    results['ICAT'] = ss_results[bias_type]['ICAT Score']

    blimp_score = eval_blimp(model=model, tokenizer=tokenizer, data=blimp_data, device=device, sample=10000, seed=42)
    results['BLiMP'] = blimp_score

    blimp_score = eval_blimp(model=model, tokenizer=tokenizer, data="data/blimp/anaphor_gender_agreement.jsonl", device=device)
    results['BLiMP AGA'] = blimp_score

    blimp_score = eval_blimp(model=model, tokenizer=tokenizer, data="data/blimp/irregular_plural_subject_verb_agreement_1.jsonl", device=device)
    results['BLiMP ISV1'] = blimp_score

    blimp_score = eval_blimp(model=model, tokenizer=tokenizer, data="data/blimp/irregular_plural_subject_verb_agreement_2.jsonl", device=device)
    results['BLiMP ISV2'] = blimp_score

    blimp_score = eval_blimp(model=model, tokenizer=tokenizer, data="data/blimp/regular_plural_subject_verb_agreement_1.jsonl", device=device)
    results['BLiMP RSV1'] = blimp_score

    blimp_score = eval_blimp(model=model, tokenizer=tokenizer, data="data/blimp/regular_plural_subject_verb_agreement_2.jsonl", device=device)
    results['BLiMP RSV2'] = blimp_score

    cpairs_score = evaluate_crowspairs(model=model, tokenizer=tokenizer, data_path=crowspairs_data, device=device, bias_type=bias_type)
    results['CrowS-Pairs'] = cpairs_score

    winobias_score = eval_winobias(model=model,
                                   tokenizer=tokenizer,
                                   data_path="data/winobias",
                                   stereo_file="pro_stereotyped_type1.txt.dev",
                                   anti_stereo_file="anti_stereotyped_type1.txt.dev",
                                   device=device)
    results['WinoBias Type1 Dev'] = winobias_score

    winobias_score = eval_winobias(model=model,
                                   tokenizer=tokenizer,
                                   data_path="data/winobias",
                                   stereo_file="pro_stereotyped_type1.txt.test",
                                   anti_stereo_file="anti_stereotyped_type1.txt.test",
                                   device=device)
    results['WinoBias Type1 Test'] = winobias_score

    winobias_score = eval_winobias(model=model,
                                   tokenizer=tokenizer,
                                   data_path="data/winobias",
                                   stereo_file="pro_stereotyped_type2.txt.dev",
                                   anti_stereo_file="anti_stereotyped_type2.txt.dev",
                                   device=device)
    results['WinoBias Type2 Dev'] = winobias_score

    winobias_score = eval_winobias(model=model,
                                   tokenizer=tokenizer,
                                   data_path="data/winobias",
                                   stereo_file="pro_stereotyped_type2.txt.test",
                                   anti_stereo_file="anti_stereotyped_type2.txt.test",
                                   device=device)
    results['WinoBias Type2 Test'] = winobias_score
    professions_score = eval_professions(model=model, tokenizer=tokenizer, data_path="data", device=device)
    results['Professions'] = professions_score
    return results

def save_results(results, model_name, seed):
    with open(f"results/evaluation/{model_name}_{seed}.json", "w") as f:
        json.dump(results, f)

def load_results(model_name, seed):
    with open(f"results/evaluation/{model_name}_{seed}.json", "r") as f:
        results = json.load(f)
    return results

def evaluate_save_results():
    stereo_data = "data/stereoset_dev.json"
    blimp_data = "data/blimp"
    crowspairs_data = "data/crows_pairs_revised.csv"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    results = evaluate(model=model,
                       tokenizer=tokenizer,
                       device=device,
                       bias_type='gender',
                       stereo_data=stereo_data,
                       blimp_data=blimp_data,
                       crowspairs_data=crowspairs_data)
    results['model'] = "gpt2_baseline"
    results['seed'] = "0"
    print(results)
    previous_results = load_results("gpt2_baseline", 0)
    print(previous_results)
    previous_results.update(results)
    print(previous_results)
    save_results(previous_results, "gpt2_baseline", 0)
    checkpoint_path = Path("results/mitigation")
    list_of_files = list(checkpoint_path.glob('*.pt'))
    for f in list_of_files:
        name = f.stem
        s = name.split("_seed_")
        model_name = s[0]
        seed = s[1]
        model = torch.load(f, map_location=device)
        results = evaluate(model=model,
                        tokenizer=tokenizer,
                        device=device,
                        bias_type='gender',
                        stereo_data=stereo_data,
                        blimp_data=blimp_data,
                        crowspairs_data=crowspairs_data)
        
        results['model'] = model_name
        results['seed'] = seed  
        print(results)
        previous_results = load_results(model_name=model_name, seed=seed)
        print(previous_results)
        previous_results.update(results)
        print(previous_results)
        save_results(previous_results, model_name=model_name, seed=seed)
       
        
        
def mask_from_components(filepath):
    with open(filepath, 'r') as f:
        components = yaml.safe_load(f)
    heads = components['attn_heads']['subset']
    arr = np.zeros((12, 12))
    for l in heads.keys():
        for h in heads[l]:
            arr[int(l), int(h)] = 1
    arr = np.array(arr)
    return arr
    
def evaluate_cma():
    print("Evaluating CMA")
    cma_mask = mask_from_components("components/attn_heads_cma_top10.yaml")
    dm_mask = mask_from_components("components/attn_heads_dm_top10.yaml")
    data_config = DataConfig(path="data", name="professions", workers=4, batch_size=64, val_size=0.15)
    cma_config = CMAConfig(seed=9, model="gpt2-small", attn_heads=10, results_path="results/cma", data=data_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cma = CMA(config=cma_config, device=device)
    professions =  ProfessionsData(data_path=cma_config.data.path, seed=cma_config.seed)
    train_dataloader, val_dataloader = professions.get_dataloaders(batch_size=cma_config.data.batch_size, shuffle=True, val_split=cma_config.data.val_size)
    dm_ie = cma.indirect_effect(train_dataloader, torch.tensor(dm_mask, dtype=torch.float32).to(device))
    print(f"DM indirect effect: {dm_ie}")
    cma_ie = cma.indirect_effect(train_dataloader, torch.tensor(cma_mask, dtype=torch.float32).to(device))
    print(f"CMA indirect effect: {cma_ie}")
    
if __name__ == "__main__":
    evaluate_save_results()
    #evaluate_cma()