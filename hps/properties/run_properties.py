from dataclasses import dataclass
import numpy as np
from hps.functions import Function
from .domain_properties import DomainProperties
from .mesh_properties import MeshProperties
from .physics_properties import PhysicsProperties


@dataclass
class RunProperties:
    domain: DomainProperties
    mesh: MeshProperties
    physics: PhysicsProperties

    # parameters
    frequencies: np.array
    top_boundary: Function
    top_parameters: np.array
    right_boundary: Function
    right_parameters: np.array
