import json
import pathlib

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from tqdm import tqdm

from hps.mesh import MeshBuilder, get_mesh
from hps.properties import RunProperties
from hps.utils import dataclass_to_dict

from dolfinx.fem.petsc import LinearProblem  # isort: skip


class Helmholtz:
    def __init__(self, properties: RunProperties):
        self.properties = properties
        self.progress = None

    def solve_subset(
        self,
        indices: np.ndarray,
        msh_path: pathlib.Path,
        out_dir: pathlib.Path,
        properties: RunProperties,
    ):
        # mesh
        msh, ct, ft = get_mesh(msh_path, comm=MPI.COMM_SELF)
        ds = ufl.Measure("ds", msh, subdomain_data=ft)
        tdim = msh.topology.dim
        top_tol = properties.domain.box_lengths[1] / 10
        top_cells = dolfinx.mesh.locate_entities(
            msh, tdim, lambda x: x[1] + top_tol >= properties.domain.box_lengths[1]
        )
        right_tol = properties.domain.box_lengths[0] / 10
        right_cells = dolfinx.mesh.locate_entities(
            msh,
            tdim,
            lambda x: x[0] + right_tol >= properties.domain.box_lengths[0],
        )

        # variational problem
        v = dolfinx.fem.FunctionSpace(
            msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2)
        )
        v_plot = dolfinx.fem.FunctionSpace(
            msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
        )
        p = ufl.TrialFunction(v)
        xi = ufl.TestFunction(v)
        p_sol = dolfinx.fem.Function(v)
        p_sol.name = "p"
        y_top = dolfinx.fem.Function(v)
        y_right = dolfinx.fem.Function(v)

        # physics constants
        v0 = 1e-3
        s = 1j * properties.physics.rho * properties.physics.c

        # initialize out file and writer
        out_file = out_dir.joinpath(f"solution_{min(indices)}.xdmf")
        writer = dolfinx.io.XDMFFile(
            MPI.COMM_SELF, out_file, "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5
        )
        writer.write_mesh(msh)

        # solve different systems
        for idx in indices:
            # get parameters
            top_params = properties.top_samples[idx]
            right_params = properties.right_samples[idx]
            f = properties.frequency_samples[idx]

            # derive physics parameters
            omega = 2 * np.pi * f
            k = omega / properties.physics.c
            ks = k**2
            y_top.interpolate(
                lambda x: properties.top_boundary(top_params)(x)
                / (properties.physics.rho * properties.physics.c),
                cells=top_cells,
            )
            y_right.interpolate(
                lambda x: properties.right_boundary(right_params)(x)
                / (properties.physics.rho * properties.physics.c),
                cells=right_cells,
            )

            # setup problem
            lhs = (
                (ufl.inner(ufl.grad(p), ufl.grad(xi)) * ufl.dx)
                - (ks * ufl.inner(p, xi) * ufl.dx)
                - (s * k * p * ufl.inner(y_top, xi) * ds(properties.mesh.top_boundary))
                - (
                    s
                    * k
                    * p
                    * ufl.inner(y_right, xi)
                    * ds(properties.mesh.right_boundary)
                )
            )
            rhs = s * k * ufl.inner(v0, xi) * ds(properties.mesh.excitation_boundary)
            # compute solution
            problem = LinearProblem(
                lhs,
                rhs,
                u=p_sol,
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "cholesky",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )
            problem.solve()

            # write solution
            out_function = dolfinx.fem.Function(v_plot)
            out_function.interpolate(p_sol)
            writer.write_function(out_function, idx)

            self.progress.update()

        # Close writer
        writer.close()

    def __call__(self, out_dir: pathlib.Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        # write run properties
        description_file = out_dir.joinpath("properties.json")
        with open(description_file, "w") as file_handle:
            json.dump(dataclass_to_dict(self.properties), file_handle)

        # mesh
        msh_path = out_dir.joinpath("mesh.msh")
        msh_builder = MeshBuilder(self.properties)
        msh_builder(msh_path)

        # solve
        self.progress = tqdm(total=self.properties.n_observations)
        self.solve_subset(
            np.arange(self.properties.n_observations),
            msh_path,
            out_dir,
            self.properties,
        )
        self.progress = None
