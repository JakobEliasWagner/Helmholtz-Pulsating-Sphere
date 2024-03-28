from dataclasses import dataclass


@dataclass
class MeshProperties:
    elements_per_wavelengths: float = 10.0
    elements_per_radians: float = 30.0

    # indices
    excitation_boundary: int = 101
    top_boundary: int = 102
    right_boundary: int = 103
