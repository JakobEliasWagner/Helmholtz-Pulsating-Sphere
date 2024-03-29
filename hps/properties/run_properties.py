from dataclasses import dataclass, field

import numpy as np

from hps.functions import Function, UniformSampler

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
    frequency_sampler: UniformSampler
    top_boundary: Function
    top_sampler: UniformSampler
    right_boundary: Function
    right_sampler: UniformSampler

    # derived parameters
    frequency_samples: np.array = field(init=False)
    top_samples: np.array = field(init=False)
    right_samples: np.array = field(init=False)

    def __post_init__(self):
        self.frequency_samples = self.frequency_sampler(n_samples=self.n_observations)
        self.top_samples = self.top_sampler(n_samples=self.n_observations)
        self.right_samples = self.right_sampler(n_samples=self.n_observations)
