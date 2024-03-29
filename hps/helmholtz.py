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

    def __call__(self, out_dir: pathlib.Path):
        # write run properties
        description_file = out_dir.joinpath("properties.json")
        with open(description_file, "w") as file_handle:
            json.dump(dataclass_to_dict(self.properties), file_handle)

        # mesh
        msh_path = out_dir.joinpath("mesh.msh")
        msh_builder = MeshBuilder(self.properties)
        msh_builder(msh_path)

        # get mesh into fenics
        msh, ct, ft = get_mesh(msh_path, comm=MPI.COMM_SELF)

        # function spaces
        v = dolfinx.fem.FunctionSpace(
            msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2)
        )
        v_plot = dolfinx.fem.FunctionSpace(
            msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
        )

        # define domain parameters
        v0 = 1e-3

        ds = ufl.Measure("ds", msh, subdomain_data=ft)
        tdim = msh.topology.dim
        top_tol = self.properties.domain.box_lengths[1] / 10
        top_cells = dolfinx.mesh.locate_entities(
            msh, tdim, lambda x: x[1] + top_tol >= self.properties.domain.box_lengths[1]
        )
        right_tol = self.properties.domain.box_lengths[0] / 10
        right_cells = dolfinx.mesh.locate_entities(
            msh,
            tdim,
            lambda x: x[0] + right_tol >= self.properties.domain.box_lengths[0],
        )

        # Define variational problem
        p = ufl.TrialFunction(v)
        xi = ufl.TestFunction(v)

        p_sol = dolfinx.fem.Function(v)
        p_sol.name = "p"

        y_top = dolfinx.fem.Function(v)
        y_right = dolfinx.fem.Function(v)

        # derived physics constants
        s = 1j * self.properties.physics.rho * self.properties.physics.c

        # Start writer
        filename = out_dir.joinpath("solution.xdmf")
        writer = dolfinx.io.XDMFFile(
            MPI.COMM_SELF, filename, "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5
        )
        writer.write_mesh(msh)

        pbar = tqdm(total=self.properties.n_observations)
        for i, (top_param, right_param, f) in enumerate(
            zip(
                self.properties.top_samples,
                self.properties.right_samples,
                self.properties.frequency_samples,
            )
        ):
            # interpolate boundaries
            y_top.interpolate(
                lambda x: self.properties.top_boundary(top_param)(x), cells=top_cells
            )
            y_right.interpolate(
                lambda x: self.properties.right_boundary(right_param)(x),
                cells=right_cells,
            )

            # setup physics
            wave_length = self.properties.physics.c / f
            k = 2 * np.pi / wave_length
            ks = k**2

            # setup problem
            lhs = (
                ufl.inner(ufl.grad(p), ufl.grad(xi)) * ufl.dx
                - (ks * ufl.inner(p, xi) * ufl.dx)
                - (
                    s
                    * k
                    * y_top
                    * ufl.inner(p, xi)
                    * ds(self.properties.mesh.top_boundary)
                )
                - (
                    s
                    * k
                    * y_right
                    * ufl.inner(p, xi)
                    * ds(self.properties.mesh.right_boundary)
                )
            )
            rhs = (
                s * k * ufl.inner(v0, xi) * ds(self.properties.mesh.excitation_boundary)
            )
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
            writer.write_function(out_function, i)

            pbar.update()

        # Close writer
        writer.close()
