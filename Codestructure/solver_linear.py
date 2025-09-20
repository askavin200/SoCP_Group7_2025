# --- Imports ---
from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import typing

import ufl
from dolfinx.io import XDMFFile

from frontend_dolfinx.KernelData import KernelData
from frontend_dolfinx.NeumannBC import NeumannBC
from assembly_linear import assemble_system, apply_boundary_conditions
from utils import ProjectStress


# --- Solver based on newton procedure ---
def solve_system(
    kernel_data: KernelData,
    dt: float,
    n_dt: int,
    matvec: list,
    theta_0: float,
    dirichlet_dofs: typing.Optional[typing.List[np.ndarray]] = None,
    dirichlet_values: typing.Optional[typing.List[np.ndarray]] = None,
    neumann_bc: typing.Optional[NeumannBC] = None,
    i: int = None,
    load_case = None
):
    # check if neumann bcs are required
    if neumann_bc is None:
        assemble_neumann = False
    else:
        assemble_neumann = True

    # initialize solution eqs
    delta_u = np.zeros(kernel_data.get_ndofs_glob(), dtype=np.float64)

    # initialize time
    time = 0.0

    if i is not None:
        cum_iter = 0

    # initialize stress projection
    def stress_ufl(u, theta, lmbda, mu, beta, theta_0):
        return (
            2 * mu * ufl.sym(ufl.grad(u))
            + lmbda * ufl.div(u) * ufl.Identity(2)
            - beta * (theta - theta_0) * ufl.Identity(2)
        )

    beta = matvec[2] * (2 * matvec[1] + 3 * matvec[0])
    stress_projector = ProjectStress(kernel_data.V, matvec[0], matvec[1], beta, theta_0, stress_ufl)

    # initialize Paraview Export
    if i is not None:
        outfile = XDMFFile(MPI.COMM_WORLD, f"Results/Cantilever/Linear/Order 2/CantileverBeam_Linear_{i}.xdmf", "w")
    else:
        outfile = XDMFFile(MPI.COMM_WORLD, f"Results/Brake_Disc/Linear/Export-brake-disc-lin-{load_case}-load.xdmf", "w")
    outfile.write_mesh(kernel_data.msh)

    # solve newton step for each time step
    for nt in range(1, n_dt + 1):
        # update time
        time += dt

        # newton procedure
        n_newton = 1

        while n_newton < 10:
            # assemble system
            A, L = assemble_system(kernel_data, dt, matvec, theta_0)

            # assemble neumann bcs
            if assemble_neumann:
                neumann_values = neumann_bc.evaluate_neumann_bc(time)
            else:
                neumann_values = None

            # apply boundary conditions
            apply_boundary_conditions(A, L, kernel_data.U.x.array, dirichlet_dofs, dirichlet_values, neumann_values)

            # solve equation system
            A = sp.csr_matrix(A)
            delta_u[:] = splinalg.spsolve(A, L)

            # relative solution error
            residual = np.linalg.norm(delta_u[:])

            if i is not None:
                cum_iter +=1

            print("Phys. time: {}, Newton Iteration: {}, Residual: {}".format(time, n_newton, residual))

            # update solution
            kernel_data.U.x.array[:] += delta_u[:]

            # check convergence
            if residual < 1e-6:
                break
            else:
                n_newton += 1

        # Evaluate stresses
        stress_projector.evaluate_stress(kernel_data.U)

        # Export solution
        field_u = kernel_data.U.sub(0).collapse()
        field_u.name = "displacement"
        outfile.write_function(field_u, time)
        field_t = kernel_data.U.sub(1).collapse()
        field_t.name = "temperature"
        outfile.write_function(field_t, time)
        outfile.write_function(stress_projector.stress, time)

        # check convergence
        if residual > 1e-6:
            outfile.close()
            raise RuntimeError("Newton procedure did not converge!")

        # update history data
        kernel_data.Un.x.array[:] = kernel_data.U.x.array[:]

    outfile.close()
    print("Calculation finished!")
    
    if i is not None:
        return np.max(kernel_data.U.x.array[:]), cum_iter