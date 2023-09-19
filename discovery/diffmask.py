import torch
import lightning.pytorch as pl

from configuration.diffmask import DiffMaskConfig 

from transformers import get_constant_schedule_with_warmup, get_constant_schedule


class DiffMask(pl.LightningModule):
    def __init__(self, config: DiffMaskConfig) -> None:
        super().__init__()
        self.config = config
        self.lambda1 = torch.nn.Parameter(torch.ones((1,)), requires_grad=True)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                params=[self.location],
                lr=self.config.trainer.lr,
            ),
            torch.optim.Adam(
                params=[self.lambda1],
                lr=self.config.trainer.lr,
            ),
        ]

        schedulers = [
            get_constant_schedule(optimizers[0]),
            get_constant_schedule(optimizers[1]),
        ]
        return optimizers, schedulers


    def optimizer_step(
        self,
        optimizer,
        optimizer_idx,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            self.lambda1.grad *= -1
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            self.lambda1.data = torch.where(
                self.lambda1.data < 0,
                torch.full_like(self.lambda1.data, 0),
                self.lambda1.data,
            )
            self.lambda1.data = torch.where(
                self.lambda1.data > 200,
                torch.full_like(self.lambda1.data, 200),
                self.lambda1.data,
            )

@torch.distributions.kl.register_kl(
    torch.distributions.Bernoulli, torch.distributions.Bernoulli
)
def kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (torch.log(p.probs + 1e-5) - torch.log(q.probs + 1e-5))
    t2 = (1 - p.probs) * (torch.log1p(-p.probs + 1e-5) - torch.log1p(-q.probs + 1e-5))
    return t1 + t2