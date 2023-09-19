import torch
import random
import logging
import hydra
import json
import numpy as np
import lightning.pytorch as pl

from pathlib import Path
from util.distributions import BinaryConcrete, RectifiedStreched
from discovery.attention_diffmask import AttentionDiffMask
from data.professions import ProfessionsData
from data.bug import BUGBalanced
from configuration.diffmask import DiffMaskConfig


from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


cs = ConfigStore.instance()
cs.store(name="diffmask_config", node=DiffMaskConfig)
log = logging.getLogger(name="main")


def results_file(config: DiffMaskConfig):
    """Get the results file name for the given configuration.
    Parameters
    ----------
    config : DiffMaskConfig
        The experiment configuration.
    Returns
    -------
    results_file_name : str
        The results file name.
    """
    results_path = Path(config.trainer.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    results_file_name = f"{config.mask.model}"
    results_file_name += f"_attn_heads_{config.mask.attn_heads}_mlps_{config.mask.mlps}"
    results_file_name += f"_epochs_{config.trainer.epochs}_lr_{config.trainer.lr}_seed_{config.seed}.json"
    return results_path.joinpath(results_file_name)

def set_seed(seed: int) -> None:
    """Set the seed for reproducibility.
    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the device to use.
    Returns
    -------
    device : torch.device
        The device to use.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_json(d, filepath):
    """Save the dictionary to the given file path.
    Parameters
    ----------
    d : dict
        The dictionary to save.
    filepath : str
        The file path to save the dictionary to.
    """
    with open(filepath, "w") as f:
        json.dump(d, f)


def expected_mask(location):
    logits = torch.tensor(location)
    dist = RectifiedStreched(BinaryConcrete(torch.full_like(logits, 0.2), logits),
                             l=-0.2,
                             r=1.0)
    return dist.expected_L0().tolist()

@hydra.main(config_path="configuration", config_name="diffmask", version_base=None)
def main(config: DiffMaskConfig) -> None:
    set_seed(config.seed)
    device = get_device()
    log.info(f"Using device: {device}")

    professions =  ProfessionsData(data_path=config.data.path, seed=config.data.seed)
    train_dataloader, val_dataloader = professions.get_dataloaders(batch_size=config.trainer.batch_size, shuffle=True, val_split=config.data.val_size)
    dm = AttentionDiffMask(config=config, device=device)
    trainer = pl.Trainer(max_epochs=config.trainer.epochs, devices=1, accelerator="auto")
    trainer.fit(dm, train_dataloader, val_dataloader)

    location = dm.location.detach().cpu().tolist()
    save_json({"location": location, "mask": expected_mask(location=location)}, results_file(config))
        

if __name__ == '__main__':
    main()
    
    