from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from hps.functions import Function, Sampler

from .domain_properties import DomainProperties
from .mesh_properties import MeshProperties
from .physics_properties import PhysicsProperties


@dataclass
class RunProperties:
    domain: DomainProperties
    mesh: MeshProperties
    physics: PhysicsProperties

    # parameters
    n_observations: int
    frequency_sampler: Sampler
    top_boundary: Function
    top_sampler: Sampler
    right_boundary: Function
    right_sampler: Sampler

    # derived parameters
    frequency_samples: np.array = field(init=False)
    top_samples: np.array = field(init=False)
    right_samples: np.array = field(init=False)

    def __post_init__(self):
        # initial samples
        frequency_samples = self.frequency_sampler(n_samples=self.n_observations)
        top_samples = self.top_sampler(n_samples=self.n_observations)
        right_samples = self.right_sampler(n_samples=self.n_observations)

        if self.n_observations == 1:
            self.top_samples = top_samples.reshape(1, 2)
            self.right_samples = right_samples.reshape(1, 2)
            self.frequency_samples = frequency_samples.reshape(
                1,
            )
            return

        # preprocess initial (unscaled dimensionality leads to inf distance)
        fs = (frequency_samples.reshape(-1, 1) - np.min(frequency_samples)) / (
            np.max(frequency_samples) - np.min(frequency_samples)
        )
        ts = (top_samples - np.min(top_samples)) / (
            np.max(top_samples) - np.min(top_samples)
        )
        rs = (right_samples - np.min(right_samples)) / (
            np.max(right_samples) - np.min(right_samples)
        )

        # sort values to speedup computation
        samples = np.concatenate([fs, ts, rs], axis=1)
        # Calculate the pairwise distances between all rows
        distances = cdist(samples, samples, metric="euclidean")
        # Set large diagonal to avoid selecting the same point
        np.fill_diagonal(distances, np.inf)
        # Start with the first point, could be any point as a starting point
        sorted_indices = [0]
        while len(sorted_indices) < self.n_observations:
            last_index = sorted_indices[-1]
            # Find the closest point that hasn't been visited
            closest_next_index = np.argmin(distances[last_index])
            # Update the distances to ensure the same point is not chosen again
            distances[:, last_index] = np.inf
            sorted_indices.append(closest_next_index)

        self.top_samples = top_samples[sorted_indices]
        self.right_samples = right_samples[sorted_indices]
        self.frequency_samples = frequency_samples[sorted_indices]
