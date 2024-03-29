from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Sampler(ABC):
    def __init__(self, min_x: Union[List, np.array], max_x: Union[List, np.array]):
        if isinstance(min_x, np.ndarray):
            self.min_x = min_x
        else:
            self.min_x = np.array(min_x, dtype=np.float64)
        if isinstance(max_x, np.ndarray):
            self.max_x = max_x
        else:
            self.max_x = np.array(max_x, dtype=np.float64)
        self.delta_x = self.max_x - self.min_x
        self.ndim = len(self.max_x)

    @abstractmethod
    def __call__(self, n_samples: int) -> np.array:
        """samples from sample domain.

        :param n_samples:
        :return:
        """
