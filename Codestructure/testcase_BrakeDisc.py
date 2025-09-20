# --- Imports ---
from mpi4py import MPI
import numpy as np
import gmsh
import dolfinx.io as dio
from petsc4py import PETSc
import dolfinx.nls as dnls

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from frontend_dolfinx.KernelData import KernelData
from frontend_dolfinx.NeumannBC import NeumannBC
from solver import solve_system as solve_system_nonlin
from solver_linear import solve_system as solve_system_lin


# --- Material parameter
E = 210e3
nu = 0.33
# Plane strain conditions
lmbda = E*nu/((1+nu)*(1-2*nu)) # First Lame parameter
mu = E/(2*(1+nu)) # second Lame parameter

# --- Case setup ---
# --- Material parameter
matpar = {"lhS": lmbda, "mhS": mu, "rho": 7.89e-9, "c_t": 452e6, "k_t": 48, "alpha_t": 11e-6}
matvec = [matpar["lhS"], matpar["mhS"], matpar["alpha_t"], matpar["rho"], matpar["c_t"], matpar["k_t"]]

"""Mesh generation and Import"""

def generate_brakedisc(MESH_SIZE: float):
    # ----- Create Geometry -------
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity",2)
    # Geometry name
    gmsh.model.add("Brake Surface")

#Points for geometry in  specific order
    # Add points in mm
    points_to_add = [[60,32.468],\
                    [60,27.468],\
                    [75,27.468],\
                    [85,2.5],\
                    [85,-2.5],\
                    [95,-2.5],\
                    [100,-3],\
                    [100,-5],\
                    [135,-5],\
                    [135,5],\
                    [100,5],\
                    [100,3],\
                    [95,2.5],\
                    [90,2.5],\
                    [80,27.468],\
                    [80,32.468],\
                    ]
    points = [gmsh.model.occ.add_point(points_to_add[i][0],points_to_add[i][1],0.0,1.0) for i in range(len(points_to_add))]
#creating boundaries using the above points (useful for applying bC)
    # Add Lines
    boundaries = [gmsh.model.occ.add_line(points[i],points[i+1]) for i in range(len(points)-1)]
    boundaries.append(gmsh.model.occ.add_line(points[-1], points[0]))

    # Create a curve loop
    surface_loop = gmsh.model.occ.add_curve_loop([boundaries[i] for i in range(len(boundaries))])

    # Create a surface
    surface = gmsh.model.occ.add_plane_surface([surface_loop])

    gmsh.model.occ.synchronize()

    # Start from last boundary? - check
    i=-1
    # Add physical groups for boundary conditions
    for i in range(len(boundaries)):
        gmsh.model.addPhysicalGroup(1, [boundaries[i]], i+1)
##Geometry Dimension - 2D
    gdim = 2
    gmsh.model.addPhysicalGroup(gdim, [surface], i+2)

    # Generate uniform mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
    gmsh.model.mesh.generate(gdim)
    # gmsh.write("BrakeDisc.inp")

    # Gmesh to DolfinX mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, cell_markers, facet_function = dio.gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

    # Define Mathematical Operations in Axisymmetric domain
    x = ufl.SpatialCoordinate(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_function)
    dvol = x[0]*ufl.dx  ## r=x[0] --> Converting dxdydz to rdrdphidz

    return mesh, cell_markers, facet_function, x, ds, dvol

# --- Parameters ---
MESH_SIZE = 1.0 # define mesh_size in mm

# Create rectangle mesh
mesh, cell_markers, facet_function, x, ds, dvol = generate_brakedisc(MESH_SIZE)

"""Definition of function space and FE-function"""
# --- Paramners ---
elmt_order = 1
solver = "Linear" # "Linear" "Nonlinear"
load_case = "Thermomechanical" # "Mechanical" "Thermal" 

# Define finite element
elmt_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), elmt_order) #Ansatz on Geometry(second parameter)
elmt_theta = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), elmt_order)

# Define function space
V_utheta = dfem.FunctionSpace(mesh, ufl.MixedElement(elmt_u, elmt_theta)) #Creating a mixed function space (usind 2 elements created above)
V_u = dfem.FunctionSpace(mesh, elmt_u)
V_theta = dfem.FunctionSpace(mesh, elmt_theta)

# Define function
U = dfem.Function(V_utheta)
U_n = dfem.Function(V_utheta)

# Split Solution function
u, theta = ufl.split(U)
u_n, theta_n = ufl.split(U_n)

# Create test functions
vu, vtheta = ufl.TestFunctions(V_utheta)

# Define function
_, v_to_vtheta = V_utheta.sub(1).collapse()

# Define initial temperature over the whole body
theta_0 = 20

# Asign initial temperature to initial soln of whole "function space"
U_n.x.array[v_to_vtheta] = theta_0 #Assigning to the whole body the Init Temp


""" Boundary Conditions"""
bc_esnt_dofs = []
bc_esnt_values = []

# --- Dirichlet boundary conditions ---
# --- Displacement
# Surface 2: No displacement
facets = facet_function.indices[facet_function.values == 2] ##Setting BCs
dofs = dfem.locate_dofs_topological((V_utheta.sub(0), V_u), 1, facets)[0] ##Extracting corresp dofs

bc_esnt_dofs.append(dofs) ##Append into list the dof
bc_esnt_values.append(np.zeros(dofs.shape[0])) ##Append the dof values

# --- Temperature
# Surface 1: Fix theta = theta_0
facets = facet_function.indices[facet_function.values == 1] ##Dirichlet DOF on Face 1
dofs = dfem.locate_dofs_topological((V_utheta.sub(1), V_theta), 1, facets)[0] ##Temp values for corresp dof

bc_esnt_dofs.append(dofs)
bc_esnt_values.append(theta_0 * np.ones(dofs.shape[0]))

# --- Neumann boundary conditions ---
def time_function(time):
    if time <= 5:
        return 1.0
    else:
        return 0
    
neumann_bc = NeumannBC(V_utheta, facet_function, time_function)
# set_traction( [value in x Direction, value in y Direction], [facet id(s)])
if load_case == "Thermomechanical":
    neumann_bc.set_traction([0, 10e3], [8]) # Pressure as a vector on Facet 8
    neumann_bc.set_traction([0, -10e3], [10]) # Pressure on Facet 10 as a vector
    neumann_bc.set_heatflux(-2370, [8, 10]) # Heat flux on Face 8 and 10
elif load_case == "Mechanical":
    neumann_bc.set_traction([0, 10e3], [8]) # Pressure as a vector on Facet 8
    neumann_bc.set_traction([0, -10e3], [10]) # Pressure on Facet 10 as a vector
elif load_case == "Thermal":
    neumann_bc.set_heatflux(-2370, [8, 10]) # Heat flux on Face 8 and 10

# --- Initialise KernelData ---
kernel_data = KernelData(V_utheta)

# Initialise the temperature history field
_, v_to_vtheta = V_utheta.sub(1).collapse()
kernel_data.Un.x.array[v_to_vtheta] = theta_0

kernel_data.set_quadrature_rule(elmt_order)
kernel_data.tabulate_shpfkt()
kernel_data.init_isomap()

"""Body Forces"""
# initialise volumetric loads
b = dfem.Function(V_u)
# b0 = dfem.Function(V_u)
r = dfem.Function(V_theta)

class Bodyforce:
    def __init__(self, rho, omega):
        self.rho = rho
        self.omega = omega
        
    def __call__(self, x):
        val = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        val[0] = self.rho * x[0] * self.omega * self.omega
        val[1] = 0
        
        return val
    
def heatsource(x):
    return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

rho = matpar["rho"]
omega_val = 189.125
bodyforce = Bodyforce(rho, omega_val)
b.interpolate(bodyforce)
r.interpolate(heatsource)
# Set centrifugal force to zero
# b0.interpolate(Bodyforce(rho, 0))
kernel_data.set_volumeforces(b)
kernel_data.set_heatsource(r)

# Solve system
if solver == "Nonlinear":
    solve_system_nonlin(kernel_data, 0.2, 50, matvec, theta_0, bc_esnt_dofs, bc_esnt_values, neumann_bc, load_case)
elif solver == "Linear":
    solve_system_lin(kernel_data, 0.2, 50, matvec, theta_0, bc_esnt_dofs, bc_esnt_values, neumann_bc, load_case)