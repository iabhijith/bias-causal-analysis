import torch
import random
import logging
import hydra
import json
import yaml
import numpy as np
import lightning.pytorch as pl

from pathlib import Path
from data.bug import BUGBalanced
from configuration.tuner import MitigationConfig
from mitigation.tuner import GPT2FineTuningModule

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


cs = ConfigStore.instance()
cs.store(name="mitigation_config", node=MitigationConfig)

LOGGER = logging.getLogger(__name__)

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

@hydra.main(config_path="configuration", config_name="tuner", version_base=None)    
def fine_tune(config: MitigationConfig):
    set_seed(config.seed)
    device = get_device()
    LOGGER.info(f"Using device: {device}")
   
    bug = BUGBalanced(data_path=config.data.path, val_split=config.data.val_size, seed=config.seed)
    train_dataloader, val_dataloader = bug.get_dataloaders(batch_size=config.tuner.batch_size)

    components_file = Path(config.model.components_path)/ (config.model.components + ".yaml")
    with open(components_file, 'r') as f:
        components = yaml.safe_load(f)
    logging.info(f"Components: {components}")
    model = GPT2FineTuningModule(config=config, components=components)
    filename = f"{config.model.name}_{config.model.components}_seed_{config.seed}"
    trainer = pl.Trainer(max_epochs=config.tuner.epochs,
                         check_val_every_n_epoch=1,
                         deterministic=True,
                         callbacks=[ModelCheckpoint(monitor='val_loss',   
                                                    mode='min',
                                                    dirpath=config.tuner.checkpoint_path,
                                                    filename=filename + "_{epoch:03d}_{val_loss:.2f}",
                                                    save_top_k=1,
                                                    every_n_epochs=1),
                                    EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  mode='min')])

    trainer.fit(model, train_dataloader, val_dataloader)
    _save_checkpoint(config, filename)

def _save_checkpoint(config: MitigationConfig, filename: str):
    """Save checkpoint to results"""
    checkpoint_path = Path(config.tuner.checkpoint_path)
    list_of_files = list(checkpoint_path.glob('*.ckpt'))
    for file in list_of_files:
        if file.stem.startswith(filename):
            LOGGER.info(f"Saving checkpoint: {file}")
            ft_model = GPT2FineTuningModule.load_from_checkpoint(file, map_location=torch.device('cpu'))
            torch.save(ft_model.model, Path(config.tuner.results_path) / (filename + ".pt"))

if __name__ == '__main__':
    fine_tune()
    
    