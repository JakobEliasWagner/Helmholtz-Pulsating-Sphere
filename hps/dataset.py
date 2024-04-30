import json
import pathlib
from dataclasses import dataclass

import numpy as np

from hps.utils import xdmf_to_numpy


@dataclass
class HelmholtzDataset:
    frequencies: np.array
    x: np.array
    p: np.array
    top_boundaries: np.array
    right_boundaries: np.array
    elements_per_wavelength: float

    @staticmethod
    def from_xdmf_file(file_path: pathlib.Path):
        # load description
        json_file = file_path.parent.joinpath("properties.json")
        with open(json_file, "r") as file_handle:
            description = json.load(file_handle)
        top_boundary = np.array(description["top_samples"])
        right_boundary = np.array(description["right_samples"])

        # load dataset
        data = xdmf_to_numpy(file_path)

        if data["Values"].ndim == 1:
            data["Values"] = data["Values"][np.newaxis, ...]

        return HelmholtzDataset(
            frequencies=data["Frequencies"],
            x=data["Geometry"],
            p=data["Values"],
            top_boundaries=top_boundary,
            right_boundaries=right_boundary,
            elements_per_wavelength=description["mesh"]["elements_per_wavelengths"],
        )
