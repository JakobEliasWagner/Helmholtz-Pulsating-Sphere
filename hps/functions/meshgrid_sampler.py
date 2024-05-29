import numpy as np

from .sampler import Sampler


class MeshGrid(Sampler):
    def __call__(self, n_samples: int) -> np.array:
        n_samples_dim = n_samples ** (1.0 / self.ndim)
        n_samples_dim = int(n_samples_dim)

        samples = np.meshgrid(
            *[
                np.linspace(self.min_x[i], self.max_x[i], n_samples_dim)
                for i in range(self.ndim)
            ]
        )
        return np.stack([sample.flatten() for sample in samples], axis=1)
