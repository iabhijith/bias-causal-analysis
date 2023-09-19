import torch
import lightning.pytorch as pl
import transformer_lens
import transformer_lens.utils as utils

from functools import partial

from typing import List
from jaxtyping import Float, Int

from .diffmask import DiffMask, kl_bernoulli_bernoulli
from util.distributions import BinaryConcrete, RectifiedStreched
from configuration.diffmask import DiffMaskConfig

from transformer_lens.hook_points import HookedRootModule, HookPoint  
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

def attention_intervention_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    counterfactual_cache: ActivationCache,
    mask: torch.Tensor,
    tail_indices: torch.Tensor,
    cf_tail_indices: torch.Tensor,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """
    Args:
        value: The attention result.
        hook: The hook point.
        counterfactual_cache: The counterfactual cache.
        mask: The mask.
        end_indices: The end indices of the sequences.
    Returns:
        The intervened attention result.
    """
    b, p, h, d = value.shape
    tail_indices = tail_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, d) 
    cf_tail_indices = cf_tail_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, d)
    counterfactual_value = counterfactual_cache[hook.name]

    v_select = torch.gather(value, 1, tail_indices)
    cf_select = torch.gather(counterfactual_value, 1, cf_tail_indices)
    mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, d)

    intervention = (1-mask) * v_select + mask * cf_select
    return torch.scatter(value, dim=1, index=tail_indices, src=intervention)

class AttentionDiffMask(DiffMask):
    def __init__(self, config: DiffMaskConfig, device):
        super().__init__(config=config)
        self.config = config
        self.automatic_optimization = False
        self.model = HookedTransformer.from_pretrained(config.mask.model, device=device)
        self.model.cfg.use_attn_result = True
        self.location = torch.nn.Parameter(torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads)), requires_grad=True)
        self.she_token = self.model.tokenizer.encode(' she')[0]
        self.he_token = self.model.tokenizer.encode(' he')[0]
       
    def intervene(self, originals, counterfactuals, mask):
        """
        Args:
            batch: A batch of original and counterfactual sequences.
            mask: A batch of masks.
        Returns:
            The logits of the original and counterfactual sequences.
        """
        cf_tokens = self.model.to_tokens(counterfactuals)
        cf_logits, counterfactual_cache, = self.model.run_with_cache(cf_tokens)
        cf_tail_indices = self._tail_indices(cf_tokens, self.model.tokenizer.eos_token_id) 
        
        o_tokens = self.model.to_tokens(originals)
        tail_indices = self._tail_indices(o_tokens, self.model.tokenizer.eos_token_id)
        o_logits = self.model(o_tokens, return_type="logits") 
        i_logits = self.model.run_with_hooks(
            o_tokens,
            return_type="logits",
            fwd_hooks=[(f"blocks.{i}.attn.hook_result",
                        partial(attention_intervention_hook,
                                counterfactual_cache=counterfactual_cache,
                                mask=mask[:, i, :].squeeze(),
                                tail_indices=tail_indices,
                                cf_tail_indices=cf_tail_indices)
                                )
                        for i in range(self.model.cfg.n_layers)],
        )
        tail_indices = tail_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, o_logits.shape[2])
        o_logits = torch.gather(o_logits, 1, tail_indices).squeeze()
        i_logits = torch.gather(i_logits, 1, tail_indices).squeeze()
        return o_logits, i_logits
    
    def _tail_indices(self, tokens, eos_token_id):
        """
        Args:
            tokens: A batch of token sequences.
            eos_token_id: The id of the end-of-sequence token.
        Returns:
            The index of the last true token in each sequence.
        """
        _, length = tokens.shape
        eos_mask = torch.where(tokens == eos_token_id, 1.0, 0.0)
        tail_indices = torch.argmax(eos_mask[:, 1:], dim=1)
        tail_indices = torch.where(tail_indices == 0, length - 1, tail_indices)
        return tail_indices
    
    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        originals, counterfactuals, y = batch
        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(self.location, 0.2), self.location), l=-0.2, r=1.0,
        )
        mask = dist.rsample(torch.Size([len(originals)]))
        expected_L0 = dist.expected_L0().sum()

        o_logits, i_logits = self.intervene(originals, counterfactuals, mask=mask)
        o_probs = torch.softmax(o_logits, dim=-1)
        i_probs = torch.softmax(i_logits, dim=-1)

        o_probs_he = o_probs[:,self.he_token].squeeze()
        o_probs_she = o_probs[:,self.she_token].squeeze()
        i_probs_he = i_probs[:,self.he_token].squeeze()
        i_probs_she = i_probs[:,self.she_token].squeeze()

        o_probs = torch.where(y == 1, o_probs_he, o_probs_she)
        i_probs = torch.where(y == 1, i_probs_she, i_probs_he)
        
        loss_r = o_probs/i_probs
        loss_r = loss_r.mean()

        loss_c = torch.distributions.kl_divergence(
                torch.distributions.Bernoulli(logits=o_logits),
                torch.distributions.Bernoulli(logits=i_logits),
            ).mean()


        loss = loss_c + loss_r +  self.lambda1 * (expected_L0 - self.config.mask.attn_heads)
       
        o1, o2 = self.optimizers()
        self.manual_backward(loss)
        self.optimizer_step(o1, 0)
        self.optimizer_step(o2, 1)

    def on_train_epoch_end(self):
        print(f"lambda1: {self.lambda1}")
        print(f"location: {self.location}")
            