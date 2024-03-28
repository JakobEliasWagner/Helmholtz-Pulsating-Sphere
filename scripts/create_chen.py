import pathlib

import dolfinx
import gmsh
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

gmsh.initialize()
gmsh.model.add("Chen")

factory = gmsh.model.occ

box = factory.add_rectangle(0, 0, 0, 0.5, 0.5)
circ = factory.add_disk(0, 0, 0, 0.05, 0.05)

factory.cut([(2, box)], [(2, circ)])

factory.synchronize()

surfs = [x[1] for x in factory.get_entities(2)]
boundaries = [x[1] for x in factory.get_entities(1)]

gmsh.model.add_physical_group(2, surfs, 1)

for boundary in boundaries:
    com = factory.get_center_of_mass(1, boundary)
    if np.isclose(com[0], 0):
        gmsh.model.add_physical_group(1, [boundary], 10)
        continue
    if np.isclose(com[0], 0.5):
        gmsh.model.add_physical_group(1, [boundary], 20)
        continue
    if np.isclose(com[1], 0):
        gmsh.model.add_physical_group(1, [boundary], 30)
        continue
    if np.isclose(com[1], 0.5):
        gmsh.model.add_physical_group(1, [boundary], 40)
        continue

    gmsh.model.add_physical_group(1, [boundary], 100)

f = 1000.0
c = 343.0
lmbda = c / f
elements_per_lmbda = 40

gmsh.option.setNumber("Mesh.MeshSizeMax", lmbda / elements_per_lmbda)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 30)

factory.synchronize()

gmsh.model.mesh.generate(2)

msh, cell_tags, boundary_tags = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0
)

gmsh.finalize()

print(set(boundary_tags.values))

# Define function space
v = dolfinx.fem.FunctionSpace(msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2))
v_plot = dolfinx.fem.FunctionSpace(
    msh, ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
)

# Define variational problem
p = ufl.TrialFunction(v)
xi = ufl.TestFunction(v)

p_sol = dolfinx.fem.Function(v)
p_sol.name = "p"

# Start writer
out_dir = pathlib.Path(__file__).parent
filename = out_dir.joinpath("sol_1000hz.xdmf")
writer = dolfinx.io.XDMFFile(
    MPI.COMM_SELF, filename, "w", encoding=dolfinx.io.XDMFFile.Encoding.HDF5
)
writer.write_mesh(msh)

k0 = 2 * np.pi / lmbda
rho0 = 1.2
s = 1j * rho0 * c
Y = 0.5 * (1 + 1j)
v_s = 1.0

ds = ufl.Measure("ds", msh, subdomain_data=boundary_tags)

# assemble problem
lhs = (
    ufl.inner(ufl.grad(p), ufl.grad(xi)) * ufl.dx
    - k0**2 * ufl.inner(p, xi) * ufl.dx
    - s * k0 * Y * ufl.inner(p, xi) * ds(20)
    - s * k0 * Y * ufl.inner(p, xi) * ds(40)
)
rhs = s * k0 * ufl.inner(v_s, xi) * ds(100)

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
writer.write_function(out_function, f)

# Close writer
writer.close()
