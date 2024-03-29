import pathlib

import gmsh
import numpy as np

from hps.properties import RunProperties


class MeshBuilder:
    def __init__(self, properties: RunProperties):
        assert (
            properties.domain.ndim == 2
        ), "This mesh builder only works in two dimensions."

        self.properties = properties
        self.f = gmsh.model.occ

    def __call__(self, out_file: pathlib.Path):
        gmsh.initialize()
        gmsh.option.setNumber(
            "General.Verbosity", 1
        )  # set verbosity level (still prints warnings)
        gmsh.model.add("Pulsating-Sphere")

        self.build_domain()
        self.set_physical_groups()
        self.set_mesh_properties()
        self.generate_mesh()
        self.save_mesh(out_file)
        gmsh.finalize()

    def build_domain(self):
        box = self.f.add_rectangle(0.0, 0.0, 0.0, *self.properties.domain.box_lengths)
        circ = self.f.add_disk(
            0.0,
            0.0,
            0.0,
            self.properties.domain.sphere_radius,
            self.properties.domain.sphere_radius,
        )

        self.f.cut([(2, box)], [(2, circ)])

    def get_groups(self) -> dict:
        groups = {}

        # surfaces
        surfs = self.f.get_entities(2)
        surfs = [s[1] for s in surfs]
        groups[2] = surfs

        # boundaries
        bl = self.properties.domain.box_lengths
        r = self.properties.domain.sphere_radius
        boundaries = {}
        top_com = np.array([bl[0] / 2, bl[1], 0.0])
        right_com = np.array([bl[0], bl[1] / 2, 0.0])
        sphere_l_com = np.array([r, r, 0.0])
        lines = self.f.get_entities(1)
        for line in lines:
            com = np.array(self.f.get_center_of_mass(*line))
            if np.allclose(com, right_com, atol=1e-2):
                boundaries[self.properties.mesh.right_boundary] = line[1]
            elif np.allclose(com, top_com, atol=1e-2):
                boundaries[self.properties.mesh.top_boundary] = line[1]
            elif np.all(np.less_equal(com, sphere_l_com)):
                # sphere
                boundaries[self.properties.mesh.excitation_boundary] = line[1]

        groups[1] = boundaries
        return groups

    def set_physical_groups(self):
        self.f.synchronize()
        groups = self.get_groups()

        # surface
        gmsh.model.add_physical_group(2, groups[2])

        # boundaries
        for idf, tag in groups[1].items():
            gmsh.model.add_physical_group(1, [tag], tag=idf)

    def set_mesh_properties(self):
        lmbda_min = self.properties.physics.c / float(
            self.properties.frequency_sampler.max_x
        )
        resolution_min = lmbda_min / self.properties.mesh.elements_per_wavelengths

        gmsh.option.setNumber("Mesh.MeshSizeMax", resolution_min)
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", self.properties.mesh.elements_per_radians
        )

    def generate_mesh(self):
        self.f.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

    def save_mesh(self, out_file: pathlib.Path):
        out_file.parent.mkdir(exist_ok=True, parents=True)
        gmsh.write(str(out_file))  # gmsh does not accept pathlib path
