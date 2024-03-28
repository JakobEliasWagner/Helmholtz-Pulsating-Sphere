from dataclasses import dataclass
from typing import List
from hps.functions import Function
from .domain_properties import DomainProperties
from .mesh_properties import MeshProperties
from .physics_properties import PhysicsProperties


@dataclass
class RunProperties:
    domain: DomainProperties
    mesh: MeshProperties
    physics: PhysicsProperties

    top_boundaries: List[Function]
    right_boundaries: List[Function]