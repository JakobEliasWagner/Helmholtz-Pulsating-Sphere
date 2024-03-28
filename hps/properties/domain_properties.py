from dataclasses import dataclass, field
import numpy as np


@dataclass
class DomainProperties:
    box_lengths: np.array = np.array([1., 1.])
    sphere_radius: float = 0.05

    ndim: int = field(init=False)

    def __post_init__(self):
        self.ndim = len(self.box_lengths)
