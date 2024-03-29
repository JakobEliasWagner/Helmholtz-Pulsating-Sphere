import numpy as np

from .sampler import Sampler


class UniformSampler(Sampler):
    def __call__(self, n_samples: int) -> np.array:
        samples = np.random.rand(n_samples, self.ndim)
        samples = samples * self.delta_x + self.min_x
        return samples.squeeze()
