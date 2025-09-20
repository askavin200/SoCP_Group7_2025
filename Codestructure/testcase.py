# --- Imports ---
from mpi4py import MPI
import numpy as np

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from frontend_dolfinx.KernelData import KernelData
from frontend_dolfinx.NeumannBC import NeumannBC
from solver import solve_system


# --- Case setup ---
# --- Material parameter
matpar = {"lhS": 60.5, "mhS": 25.9, "rho": 2699, "c_t": 888, "k_t": 235, "alpha_t": 23.1 * 1e-6}
matvec = [matpar["lhS"], matpar["mhS"], matpar["alpha_t"], matpar["rho"], matpar["c_t"], matpar["k_t"]]

# Reference temperature
theta_0 = 5.0

# --- Mesh generation
l_domain = [1, 1]
n_elmt = [10, 10]

# Create rectangle mesh
msh = dmesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([l_domain[0], l_domain[1]])],
    [n_elmt[0], n_elmt[1]],
    cell_type=dmesh.CellType.triangle,
    diagonal=dmesh.DiagonalType.left,
)

# Mark boundary elements wit tag
boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[1], 0)),
    (3, lambda x: np.isclose(x[0], l_domain[0])),
    (4, lambda x: np.isclose(x[1], l_domain[1])),
]

facet_indices, facet_markers = [], []
for marker, locator in boundaries:
    facets = dmesh.locate_entities(msh, 1, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))

facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_function = dmesh.meshtags(msh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets])
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_function)

# --- Setup function spaces
elmt_order = 1

# Define finite element
elmt_u = ufl.VectorElement("Lagrange", msh.ufl_cell(), elmt_order)
elmt_t = ufl.FiniteElement("Lagrange", msh.ufl_cell(), elmt_order)

# Define function space
V_ut = dfem.FunctionSpace(msh, ufl.MixedElement(elmt_u, elmt_t))
V_u, v_to_vu = V_ut.sub(0).collapse()
V_t, v_to_vt = V_ut.sub(1).collapse()

# --- Set dirichlet-boundary conditions
bc_esnt_dofs = []
bc_esnt_values = []

# displacement
# left (no horizontal displacement)
facets = facet_function.indices[facet_function.values == 1]
dofs = dfem.locate_dofs_topological((V_ut.sub(0).sub(0), V_u.sub(0)), 1, facets)[0]

bc_esnt_dofs.append(dofs)
bc_esnt_values.append(0 * np.ones(dofs.shape[0]))

# bottom (no vertical displacement)
facets = facet_function.indices[facet_function.values == 2]
dofs = dfem.locate_dofs_topological((V_ut.sub(0).sub(1), V_u.sub(1)), 1, facets)[0]

bc_esnt_dofs.append(dofs)
bc_esnt_values.append(0 * np.ones(dofs.shape[0]))

# temperature
# top (t = 0)
facets = facet_function.indices[facet_function.values == 4]
dofs = dfem.locate_dofs_topological((V_ut.sub(1), V_t), 1, facets)[0]

bc_esnt_dofs.append(dofs)
bc_esnt_values.append(theta_0 * np.ones(dofs.shape[0]))

# --- Set dirichlet-boundary conditions
neumann_bc = NeumannBC(V_ut, facet_function)
neumann_bc.set_traction([0, 7], [4])
neumann_bc.set_heatflux(-30000, [3])

# --- Set volumetric sources
b = dfem.Function(V_u)
r = dfem.Function(V_t)

# --- Initialise KernelData ---
kernel_data = KernelData(V_ut)
kernel_data.set_quadrature_rule(1)
kernel_data.tabulate_shpfkt()
kernel_data.init_isomap()
kernel_data.set_volumeforces(b)
kernel_data.set_heatsource(r)

# --- Perform calculation ---
# Set initial values
kernel_data.Un.x.array[v_to_vt] = theta_0

# Solve system
solve_system(kernel_data, 100, 10, matvec, theta_0, bc_esnt_dofs, bc_esnt_values, neumann_bc)
