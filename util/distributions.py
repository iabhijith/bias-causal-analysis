"""
Implementation of the binary concrete distribution and its rectified version from De Cao et al. 2021
"""
import torch
import logging

log = logging.getLogger(__name__)

class BinaryConcrete(torch.distributions.relaxed_bernoulli.RelaxedBernoulli):
    """
    Binary concrete distribution.
    """
    def __init__(self, temperature, logits):
        """
        Args:
        temperature: The temperature of the distribution.
        logits: The logits of the distribution.
        """
        super().__init__(temperature=temperature, logits=logits)
        self.device = self.temperature.device

    def cdf(self, value):
        return torch.sigmoid(
            (torch.log(value) - torch.log(1.0 - value)) * self.temperature - self.logits
        )

    def log_prob(self, value):
        return torch.where(
            (value > 0) & (value < 1),
            super().log_prob(value),
            torch.full_like(value, -float("inf")),
        )

    def log_expected_L0(self, value):
        return -torch.nn.functional.softplus(
            (torch.log(value) - torch.log(1 - value)) * self.temperature - self.logits
        )


class Streched(torch.distributions.TransformedDistribution):
    """
    Streched version of the binary concrete distribution.
    """
    def __init__(self, base_dist, l=-0.1, r=1.1):
        """
        Args:
        base_dist: The base distribution.
        l: The left boundary of the streched distribution.
        r: The right boundary of the streched distribution.
        """
        super().__init__(
            base_dist, torch.distributions.AffineTransform(loc=l, scale=r - l)
        )

    def log_expected_L0(self):
        """
        Returns the log expected L0 norm of the distribution.
        """
        value = torch.tensor(0.0, device=self.base_dist.device)
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.log_expected_L0(value)
        value = self._monotonize_cdf(value)
        return value

    def expected_L0(self):
        """ 
        Returns the expected L0 norm of the distribution.
        """
        return self.log_expected_L0().exp()


class RectifiedStreched(Streched):
    """
    Rectified version of the streched distribution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, sample_shape=torch.Size([])):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size([])):
        x = super().rsample(sample_shape)
        return x.clamp(0, 1)



if __name__ == '__main__':
    logits = torch.tensor([[-10, 0, 1000], [0, 0, 0]])
    dist = RectifiedStreched(BinaryConcrete(torch.full_like(logits, 0.2), logits),
                             l=-0.2,
                             r=1.0)
    print(dist.rsample(sample_shape=torch.Size([10])).shape)
    print(dist.expected_L0())
    print(dist.expected_L0(torch.tensor(0.0)))