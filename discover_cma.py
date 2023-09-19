import torch
import random
import logging
import hydra
import json
import numpy as np
import lightning.pytorch as pl

from pathlib import Path

from discovery.cma import CMA
from data.professions import ProfessionsData
from configuration.cma import CMAConfig


from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


cs = ConfigStore.instance()
cs.store(name="cma_config", node=CMAConfig)
log = logging.getLogger(name="main")


def results_file(config: CMAConfig):
    """Get the results file name for the given configuration.
    Parameters
    ----------
    config : CMAConfig
        The experiment configuration.
    Returns
    -------
    results_file_name : str
        The results file name.
    """
    results_path = Path(config.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    results_file_name = f"{config.model}"
    results_file_name += f"_attn_heads_{config.attn_heads}"
    results_file_name += f"_seed_{config.seed}.json"
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


@hydra.main(config_path="configuration", config_name="cma", version_base=None)
def main(config: CMAConfig) -> None:
    set_seed(config.seed)
    device = get_device()
    log.info(f"Using device: {device}")

    professions =  ProfessionsData(data_path=config.data.path, seed=config.seed)
    train_dataloader, val_dataloader = professions.get_dataloaders(batch_size=config.data.batch_size, shuffle=True, val_split=config.data.val_size)
    cma = CMA(config=config, device=device)
    indirect_effects = cma.indirect_effects(train_dataloader)
    save_json({"indirect_effects": indirect_effects.tolist()}, results_file(config))
        

if __name__ == '__main__':
    main()
    
    