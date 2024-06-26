from dataclasses import dataclass, field

import numpy as np


@dataclass
class DomainProperties:
    box_lengths: np.array = np.array([1.0, 1.0])
    sphere_radius: float = 0.2

    ndim: int = field(init=False)

    def __post_init__(self):
        self.ndim = len(self.box_lengths)
