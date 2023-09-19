import torch
import logging
import einops
import lightning.pytorch as pl

from functools import cached_property


from configuration.tuner import MitigationConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup


LOGGER = logging.getLogger(__name__)

ALL = "all"
SUBSET = "subset"
WTE = "wte"
WPE = "wpe"
LN = "ln"
MLP = "mlp"
MLPS = "mlps"
ATTN = "attn"
ATTN_LAYERS = "attn_layers"
ATTN_HEADS = "attn_heads"


class GPT2FineTuningModule(pl.LightningModule):
    def __init__(self, config: MitigationConfig, components: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel.from_pretrained(config.model.name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model.name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.config = config
        self.components = components
       
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        X, y = batch
        inputs = self.tokenizer(X, padding=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, -100, input_ids) 
        input_ids = torch.where(input_ids == self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, input_ids)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        inputs = self.tokenizer(X, padding=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, -100, input_ids) 
        input_ids = torch.where(input_ids == self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, input_ids)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

        
    def on_before_optimizer_step(self, optimizer):
        """
        Allow gradients only for the specified attention heads.
        """
        if self.attn_mask is not None:
            bias_mask = self.attn_mask.unsqueeze(-1).to(self.device)
            weights_mask = bias_mask.unsqueeze(-1)
            for n, p in self.model.named_parameters():
                if ATTN in n and p.grad is not None:
                    l = int(n.split(".")[2])
                    if "weights" in n:
                        m = weights_mask[l]
                        P_Q, P_K, P_V = torch.tensor_split(p.grad, 3, dim=1)
                        P_Q = einops.rearrange(P_Q, "m (i h)->i m h", i=self.model.config.n_head)
                        P_K = einops.rearrange(P_K, "m (i h)->i m h", i=self.model.config.n_head)
                        P_V = einops.rearrange(P_V, "m (i h)->i m h", i=self.model.config.n_head)
                        P_Q.mul_(m)
                        P_K.mul_(m)
                        P_V.mul_(m)
                    elif "bias" in n:
                        m = bias_mask[l]
                        P_Q, P_K, P_V = torch.tensor_split(p.grad, 3, dim=0)
                        P_Q = einops.rearrange(P_Q, "(i h)->i h", i=self.model.config.n_head)
                        P_K = einops.rearrange(P_K, "(i h)->i h", i=self.model.config.n_head)
                        P_V = einops.rearrange(P_V, "(i h)->i h", i=self.model.config.n_head)
                        P_Q.mul_(m)
                        P_K.mul_(m)
                        P_V.mul_(m)
        
    def configure_optimizers(self):
        """
        Configure optimizer based on the config. Enable grad for only the components specified in the configuration.
        """
        self._freeze_components()
        no_decay = ["bias", "ln"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.tuner.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.tuner.lr, eps=self.config.tuner.eps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.tuner.warmup,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def _freeze_components(self):
        """
        Freeze the components of the model that are not to be fine-tuned.
        """
        all = ALL in self.components and self.components[ALL]
        for n, p in self.model.named_parameters():
            if all or self._requires_finetuning(n):
                p.requires_grad = True
                LOGGER.info(f"Fine-tuning {n}")
            else:
                p.requires_grad = False
        

    def _requires_finetuning(self, name):
        """
        Check if the component requires fine-tuning.
        """
        all_mlps = MLPS in self.components and ALL in self.components[MLPS] and self.components[MLPS][ALL]
        all_attn_layers = ATTN_LAYERS in self.components and ALL in self.components[ATTN_LAYERS] \
            and self.components[ATTN_LAYERS][ALL]  
        if WTE in name:
            return WTE in self.components and self.components[WTE]
        elif WPE in name :
            return WPE in self.components and self.components[WPE]
        elif LN in name:
            return LN in self.components and self.components[LN]
        elif MLP in name:
            return all_mlps or any(mlp in name for mlp in self.mlps)
        elif ATTN in name:
            return all_attn_layers or any(attention_layer in name for attention_layer in self.attention_layers)
                                  
    @cached_property
    def mlps(self):
        """
        Get the MLPs to be fine-tuned.
        """
        selected_mlps = []
        if MLPS in self.components and SUBSET in self.components[MLPS]:
            selected_mlps.extend([f"h.{m}.mlp" for m in self.components[MLPS][SUBSET]])
        return selected_mlps
    
    @cached_property
    def attention_layers(self):
        """
        Get the attention layers to be fine-tuned.
        """
        if all([ATTN_HEADS in self.components, ATTN_LAYERS in self.components]):
            raise ValueError("Cannot specify attention heads and attention layers at the same time.")
        selected_attention_layers = []
        if ATTN_LAYERS in self.components and SUBSET in self.components[ATTN_LAYERS]:
            selected_attention_layers.extend([f"h.{l}.attn" for l in self.components[ATTN_LAYERS][SUBSET]])
        elif ATTN_HEADS in self.components and SUBSET in self.components[ATTN_HEADS]:
            selected_attention_layers.extend([f"h.{l}.attn.c_attn" for l in self.components[ATTN_HEADS][SUBSET]])
        return selected_attention_layers   
                    
    @cached_property
    def attn_mask(self):
        """
        Get the mask for the attention heads to be fine-tuned.
        """
        if all([ATTN_HEADS in self.components, ATTN_LAYERS in self.components]):
            raise ValueError("Cannot specify attention heads and attention layers at the same time.")
        elif ATTN_HEADS in self.components and SUBSET in self.components[ATTN_HEADS]:
            n_layer  = self.model.config.n_layer
            n_head = self.model.config.n_head
            mask = torch.zeros(n_layer, n_head)  
            for l in self.components[ATTN_HEADS][SUBSET]:
                for h in self.components[ATTN_HEADS][SUBSET][l]:
                    mask[l,h] = 1      
            return mask
        else:
            return None
        

            

        
        
