# --- Imports ---
import numpy as np
import scipy.sparse as sp
import typing

import dolfinx.fem as dfem

from frontend_dolfinx.KernelData import KernelData
from utils import interpolate, interpolate_grad, dCinvdC, convert_S_voigt, B_nonlinear, \
convert_4_order_voigt, convert_2_order_voigt, derivative_dlnJ_dC_dyad_Cinv


# --- Assembly routines on nodal level ---
# --- Tangent stiffness
def stiffness_element_uu(matpar: list, dphi_i: np.ndarray, dphi_j: np.ndarray, alpha_gp: float, theta_np1k: np.ndarray, det_j: float,
    theta0, beta, S: np.ndarray, F: np.ndarray) -> np.ndarray:
    
    # Extract material parameters
    lhS = matpar[0]
    mhS = matpar[1]

    C = np.dot(F.T,F)
    C_inv = np.linalg.inv(C) 

    B_I = B_nonlinear(F, dphi_i)
    B_J = B_nonlinear(F, dphi_j)

    dS_dC = (-0.5 * (np.einsum('ik, jl -> ijkl', C_inv, C_inv) + np.einsum('il, jk -> ijkl', C_inv, C_inv)) *
         (lhS * np.log(np.linalg.det(F)) - mhS - beta * (theta_np1k - theta0)) + lhS * 0.5 * np.einsum('ij, kl -> ijkl', C_inv, C_inv))
    dS_dC_voigt = convert_4_order_voigt(dS_dC)

    kuu_nl = (alpha_gp*det_j*(np.linalg.multi_dot([B_I.T, dS_dC_voigt, 2 * B_J]) + np.linalg.multi_dot([dphi_i.T, S, dphi_j]) * np.eye(2)))

    return kuu_nl


def stiffness_element_ut(dphi_i: np.ndarray, phi_j: np.ndarray, alpha_gp: float, det_j: float, beta, F: np.ndarray) -> np.ndarray:
    
    C = np.dot(F.T,F)
    C_inv = np.linalg.inv(C)
    C_inv_voigt = convert_2_order_voigt(C_inv)

    B_I = B_nonlinear(F, dphi_i)

    kut_nl = alpha_gp * det_j * np.dot(B_I.T, C_inv_voigt) * -beta * phi_j

    return kut_nl


def stiffness_element_tu(matpar: list, phi_i: np.ndarray, dphi_i: np.ndarray, dphi_j: np.ndarray, alpha_gp: float, det_j: float, F: np.ndarray,
    F_n: np.ndarray, dtheta: np.ndarray, dt: float, theta_np1k: np.ndarray, beta) -> np.ndarray:

    # Extract material parameters
    k_t = matpar[5]

    C = np.dot(F.T,F)
    C_inv = np.linalg.inv(C)
    C_inv_voigt = convert_2_order_voigt(C_inv)

    C_dot = np.dot(F.T, F) - np.dot(F_n.T, F_n)
    C_dot_voigt = convert_2_order_voigt(C_dot)

    DCinvdC = dCinvdC(C_inv)
    dCinvdC_voigt = convert_4_order_voigt(DCinvdC)    

    B_J = B_nonlinear(F, dphi_j)

    ke_1 = np.zeros(2)  # Ktu Part 1
    ke_1 = np.dot((phi_i * beta * 0.5 * theta_np1k * (np.dot(dCinvdC_voigt, C_dot_voigt) + C_inv_voigt)).T, 2 * B_J)

    dlnJ_dC_dyad_Cinv_Voigt = convert_4_order_voigt(derivative_dlnJ_dC_dyad_Cinv(C))

    ke_2 = np.zeros(2)  # Ktu Part 2

    Li_Gradtheta_voigt = np.array([
                        dphi_i[0] * dtheta[0,0],
                        dphi_i[1] * dtheta[0,1],
                        dphi_i[0] * dtheta[0,1] + dphi_i[1] * dtheta[0,0] ])

    ke_2 = 2 * k_t * dt * np.linalg.det(F) * Li_Gradtheta_voigt @ np.dot((dCinvdC_voigt + dlnJ_dC_dyad_Cinv_Voigt), B_J)

    ktu_nl = alpha_gp * det_j * (ke_1 + ke_2)

    return ktu_nl


def stiffness_element_tt(matpar: list, dt: float, phi_i: np.ndarray, phi_j: np.ndarray, dphi_i: np.ndarray, dphi_j: np.ndarray, alpha_gp: float,
    det_j: float, J: float, F: np.ndarray, F_n: np.ndarray, beta) -> np.ndarray:

    # Extract material parameters
    rho_0 = matpar[3]
    c_t = matpar[4]
    k_t = matpar[5]

    C = np.dot(F.T,F)
    C_inv = np.linalg.inv(C)

    C_dot = np.dot(F.T, F) - np.dot(F_n.T, F_n)

    # Assemble stiffness matrix
    ktt_nl = (alpha_gp * det_j * (phi_i * (0.5 * beta * np.tensordot(C_inv, C_dot) + rho_0 * c_t) * phi_j
            + k_t * dt * J * np.linalg.multi_dot([dphi_i.T, C_inv, dphi_j])))

    return ktt_nl


# --- Load vector
def load_u(S_voigt: np.ndarray, b: np.ndarray, phi_i: np.ndarray, dphi_i: np.ndarray, alpha_gp: np.array, det_j: float, F: np.array):

    B_nl_i = B_nonlinear(F, dphi_i)

    # calculate load vector
    lu_nl = -alpha_gp * det_j * (np.dot(B_nl_i.T, S_voigt) + phi_i * b)

    return lu_nl


def load_t(matpar: list, theta_dot_gp: np.ndarray, theta_np1k_gp: float, dt: float, dthetadx_np1k: np.ndarray, rhor_gp: np.ndarray,
    phi_i: np.ndarray, dphi_i: np.ndarray, alpha_gp: np.array, det_j: float, detF: float, F: np.array, F_n: np.array, beta):

    # Extract material parameters
    rho_0 = matpar[3]
    c_t = matpar[4]
    k_t = matpar[5] * dt

    C = np.dot(F.T,F)
    C_inv = np.linalg.inv(C)

    C_dot = np.dot(F.T, F) - np.dot(F_n.T, F_n)

    lt_nl = (-alpha_gp*det_j*((rho_0 * c_t * theta_dot_gp + beta * theta_np1k_gp * 0.5 * np.tensordot(C_dot, C_inv) - rhor_gp)
            * phi_i + k_t * detF * np.linalg.multi_dot([dthetadx_np1k, C_inv, dphi_i])))

    return lt_nl


# --- Assembly on the element level ---
def assemble_system_elmt(
    c: int,
    kernel_data: KernelData,
    dt: float,
    matvec_ufl: typing.List[dfem.Function],
    theta0: float,
):
    """Assembles right- and left-hand side of for 2D thermo-elasticity.

    Args:
        c (int): The cell index.
        kd (class KernelData): Data for assembly of the stiffness matrix.
        matvec (list): List of material parameters.
                       [first Lame constant, second Lame constant, thermal expansion coefficient, density, heat capacity, thermal conductivity]

    Returns:
        np.ndarray: The element-stiffness matrix.
    """

    # --- Get relevant values
    # Number of DOFs per element
    ndof_per_cell = kernel_data.get_ndofs_per_cell()

    # Number of nodes per element
    nodes_per_elmt = kernel_data.get_nnodes_per_cell()

    # Initialize element stiffness matrix
    ke = np.zeros((ndof_per_cell, ndof_per_cell), dtype=np.float64)
    le = np.zeros(ndof_per_cell, dtype=np.float64)

    # Material parameters
    matvec = []

    if (hasattr(matvec_ufl[0], 'x')):
        # Material parameters
        matvec2 = []
        for uflparam in matvec:
            matvec2.append(uflparam.x.array[c])
        matvec = matvec2
    else:
        matvec = matvec_ufl

    # ----------------- Calculate J, inv(J) and |det(J)| ---------------------- #
    j, det_j = kernel_data.isomap_elmt(c)
    jinv = np.linalg.inv(j)

    # ---------------- Transform Gradients to physical coordinates ------------ #
    # get quadrature weights
    alpha = kernel_data.get_quadrature_weights()

    # get tabulated shape functions (reference cell)
    phi = kernel_data.get_shape_functions()
    dphi = phi[1:, :, :, 0]

    # map derivatives to physical coordinates
    dphi_x = np.einsum("mk, mij -> kij", jinv, dphi)

    # --------------- Evaluate volume loads at qudrature points --------------- #
    # body force
    hat_b = kernel_data.get_volume_forces(c, 0)

    if hat_b is None:
        vec_b = np.zeros((alpha.shape[0], 2))
    else:
        vec_b = interpolate(hat_b, phi, bs=2)

    # volumetric heat source
    hat_r = kernel_data.get_volume_forces(c, 1)

    if hat_r is None:
        vec_rhor = np.zeros(alpha.shape[0])
    else:
        rho_0 = matvec[3]
        vec_rhor = interpolate(rho_0 * hat_r, phi, bs=1)[:, 0]

    # --------------- Evaluate solution data at qudrature points -------------- #
    # extract displacement (time n+1, iterate k) at nodes
    dofs_u = kernel_data.get_cell_solution_np1(c, subspace=0)

    # extract temperature (time n+1, iterate k) at nodes
    dofs_t = kernel_data.get_cell_solution_np1(c, subspace=1)

    # extract displacement (time n) at nodes
    dofs_u_n = kernel_data.get_cell_solution_n(c, subspace=0)

    # extract temperature (time n) at nodes
    dofs_t_n = kernel_data.get_cell_solution_n(c, subspace=1)

    # evaluate the gradient/divergence of the displacement (time n+1, iterate k) at quadrature points
    dudx_np1k = interpolate_grad(dofs_u, dphi_x, bs=2)
    # evaluate the gradient/divergence of the displacement (time n, iterate k) at quadrature points
    DuDx_n = interpolate_grad(dofs_u_n, dphi_x, bs=2)

    # evaluate temperature (time n+1, iterate k) at quadrature points
    theta_np1k = interpolate(dofs_t, phi, bs=1)[:, 0]

    # evaluate time-derivative of temperature (time n+1, iterate k) at quadrature points
    thetadot_np1k = interpolate(dofs_t - dofs_t_n, phi, bs=1)[:, 0]

    # evaluate the temperature gradient (time n+1, iterate k) at quadrature points
    dthetadx_np1k = interpolate_grad(dofs_t, dphi_x, bs=1)

    # evaluate stress (time n+1, iterate k) at quadrature points (Voigt notation!)
    lhS = matvec[0]
    mhS = matvec[1]
    alpha_t = matvec[2]
    beta = alpha_t * (2 * mhS + 3 * lhS)

    # loop over all quadrature points
    for igp in range(0, alpha.shape[0]):

        # Extract all quantities at gauss point
        DuDx_n_gp = DuDx_n[igp, :, :]

        # Calculate current deformation tensor
        F = dudx_np1k[igp,:,:] + np.eye(2)
        # Calculate previous deformation tensor
        F_n = DuDx_n_gp + np.eye(2)

        # Calculate caughy green deformation tensor
        C = np.dot(F.T, F)

        # calculate inverse of C
        C_inv = np.linalg.inv(C)

        # Calculate determinant of F
        J = np.linalg.det(F)

        # 2nd Piola Stress
        S = (mhS * np.eye((2)) + (lhS * np.log(J) - mhS - beta * (theta_np1k[igp] - theta0)) * C_inv)
        S_voigt = convert_S_voigt(S)

        # assemble stiffness and load vector per element
        for ii in range(0, nodes_per_elmt):
            # calculate local dof indices for displacement/ temperature DOFs
            iu1 = kernel_data.get_node_dofs_local(ii, 0)[0]
            iu2 = iu1 + 1
            it = kernel_data.get_node_dofs_local(ii, 1)

            # evaluate load vector contributions
            l_u = load_u(S_voigt, vec_b[igp, :], phi[0, :, ii, 0][[igp]], dphi_x[:, igp, ii].T, alpha[igp], det_j, F)
            l_t = load_t(matvec, thetadot_np1k[igp], theta_np1k[igp], dt, dthetadx_np1k[igp, :], vec_rhor[igp], phi[0, igp, ii, 0],
                         dphi_x[:, igp, ii].T, alpha[igp], det_j, J, F, F_n, beta)

            # fill the values into the elemental load vector
            le[iu1] += l_u[0]
            le[iu2] += l_u[1]
            le[it] += l_t

            for jj in range(0, nodes_per_elmt):
                # calculate local dof indices for first and second DOF
                ju1 = kernel_data.get_node_dofs_local(jj, 0)[0]
                ju2 = ju1 + 1
                jt = kernel_data.get_node_dofs_local(jj, 1)

                # evaluate local stiffness constributions
                k_uu = stiffness_element_uu(matvec, dphi_x[:, igp, ii].T, dphi_x[:, igp, jj].T, alpha[igp], theta_np1k[igp], det_j, theta0, beta, S, F)
                k_ut = stiffness_element_ut(dphi_x[:, igp, ii].T, phi[0, igp, jj, 0], alpha[igp], det_j, beta, F)
                k_tu = stiffness_element_tu(matvec, phi[0, igp, ii, 0], dphi_x[:, igp, ii].T, dphi_x[:, igp, jj].T, alpha[igp], det_j, F, F_n,
                                            dthetadx_np1k[igp,:,:], dt, theta_np1k[igp], beta)
                k_tt = stiffness_element_tt(matvec, dt, phi[0, igp, ii, 0], phi[0, igp, jj, 0], dphi_x[:, igp, ii].T, dphi_x[:, igp, jj].T, alpha[igp],
                                            det_j, J, F, F_n, beta)

                # fill the values into the elemental stiffness matrix
                # stiffness due to displacement
                ke[iu1, ju1] += k_uu[0, 0]
                ke[iu1, ju2] += k_uu[0, 1]
                ke[iu2, ju1] += k_uu[1, 0]
                ke[iu2, ju2] += k_uu[1, 1]

                # stiffness displacment/ temperature coupling
                ke[iu1, jt] += k_ut[0]
                ke[iu2, jt] += k_ut[1]

                # stiffness temperature/ displacement coupling
                ke[it, ju1] += k_tu[0]
                ke[it, ju2] += k_tu[1]

                # stiffness due to temperature
                ke[it, jt] += k_tt

    return ke, le


# --- Assembly on the global level ---
def assemble_system(
    kernel_data: KernelData,
    dt: float,
    matvec_ufl: typing.List[dfem.Function],
    theta_0: float,
):
    # counters
    number_of_cells = kernel_data.get_ncells_mesh()
    ndofs_global = kernel_data.get_ndofs_glob()

    # initialize tangents
    A = sp.lil_matrix((ndofs_global, ndofs_global))
    L = np.zeros(ndofs_global, dtype=np.float64)

    # loop over all elements
    for c in range(0, number_of_cells):
        # calculate elemental stiffness
        Ke, Le = assemble_system_elmt(c, kernel_data, dt, matvec_ufl, theta_0)

        # extract global DOFs on cell
        cell_dofs = kernel_data.get_cell_dofs(c, subspace=None)

        # assemble Ke into global system matrix
        for ii, dofi in enumerate(cell_dofs):
            L[dofi] += Le[ii]

            for jj, dofj in enumerate(cell_dofs):
                A[dofi, dofj] += Ke[ii, jj]

    return A, L


def apply_boundary_conditions(
    A: sp.lil_matrix,
    L: np.ndarray,
    U_np1_k: np.ndarray,
    dirichlet_dofs=None,
    dirichlet_values=None,
    neumann_values=None,
):
    # set neumann values
    # assumptions: 1.) neumann_values is list of numpy arrays
    #              2.) neumann_values[i].shape[0] = ndofs_global

    if neumann_values is not None:
        L[:] += neumann_values[:]

    # set boundary conditions
    # assumption: dirichlet_dofs/dirichlet_values are list of numpy arrays
    if dirichlet_dofs is not None:
        # apply dirichlet values from input
        for array_dofs, array_values in zip(dirichlet_dofs, dirichlet_values):
            for dof, value in zip(array_dofs, array_values):
                A[dof, :] = 0
                A[dof, dof] = 1
                L[dof] = value - U_np1_k[dof]
