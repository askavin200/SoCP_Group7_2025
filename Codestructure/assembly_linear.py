# --- Imports ---
import numpy as np
import scipy.sparse as sp

from frontend_dolfinx.KernelData import KernelData
from utils import interpolate, interpolate_grad


# --- Assembly routines on nodal level ---
# --- Tangent stiffness
def stiffness_element_uu(
    matpar: list, dphi_i: np.ndarray, dphi_j: np.ndarray, alpha: np.array, det_j: float
) -> np.ndarray:

    # extract material parameters
    lhS = matpar[0]
    mhS = matpar[1]

    # precalculate the products of the respective gradients (colored boxes)
    phi_xx = np.multiply(dphi_i[:, 0], dphi_j[:, 0])
    phi_yy = np.multiply(dphi_i[:, 1], dphi_j[:, 1])
    phi_yx = np.multiply(dphi_i[:, 1], dphi_j[:, 0])
    phi_xy = np.multiply(dphi_i[:, 0], dphi_j[:, 1])

    # calculate relevant entries of the material tensor
    c_11 = lhS + 2 * mhS
    c_33 = mhS
    c_12 = lhS

    # initialize the element stiffness matrix with zeros
    kuu = np.zeros((2, 2))

    # use `np.dot` with the given quadrature weights to calculate the elementwise integral
    kuu[0, 0] = np.dot(c_11 * phi_xx + c_33 * phi_yy, alpha) * det_j
    kuu[0, 1] = np.dot(c_12 * phi_xy + c_33 * phi_yx, alpha) * det_j
    kuu[1, 0] = np.dot(c_12 * phi_yx + c_33 * phi_xy, alpha) * det_j
    kuu[1, 1] = np.dot(c_33 * phi_xx + c_11 * phi_yy, alpha) * det_j

    return kuu


def stiffness_element_ut(
    matpar: list, dphi_i: np.ndarray, phi_j: np.ndarray, alpha: np.array, det_j: float
) -> np.ndarray:
    # Extract material parameters
    lhS = matpar[0]
    mhS = matpar[1]
    alpha_t = matpar[2]

    # Initialize stiffness matrix
    ke = np.zeros(2, dtype=np.float64)

    # Assemble stiffness matrix
    beta = alpha_t * (2 * mhS + 3 * lhS)

    ke[0] = -beta * np.dot(np.multiply(dphi_i[:, 0], phi_j[:]), alpha) * det_j
    ke[1] = -beta * np.dot(np.multiply(dphi_i[:, 1], phi_j[:]), alpha) * det_j

    return ke


def stiffness_element_tu(
    matpar: list, theta_0: float, phi_i: np.ndarray, dphi_j: np.ndarray, alpha: np.array, det_j: float
) -> np.ndarray:
    # Extract material parameters
    lhS = matpar[0]
    mhS = matpar[1]
    alpha_t = matpar[2]

    # Initialize stiffness matrix
    ke = np.zeros(2, dtype=np.float64)

    # Assemble stiffness matrix
    beta_theta0 = alpha_t * (2 * mhS + 3 * lhS) * theta_0

    ke[0] = beta_theta0 * np.dot(np.multiply(phi_i[:], dphi_j[:, 0]), alpha) * det_j
    ke[1] = beta_theta0 * np.dot(np.multiply(phi_i[:], dphi_j[:, 1]), alpha) * det_j

    return ke


def stiffness_element_tt(
    matpar: list,
    dt: float,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    dphi_i: np.ndarray,
    dphi_j: np.ndarray,
    alpha: np.array,
    det_j: float,
) -> np.ndarray:
    # Extract material parameters
    rho_0 = matpar[3]
    c_t = matpar[4]
    k_t = matpar[5] * dt

    # Assemble stiffness matrix
    rho_t_c = rho_0 * c_t

    ke = (
        rho_t_c * np.dot(np.multiply(phi_i[:], phi_j[:]), alpha) * det_j
        + k_t * np.dot(np.multiply(dphi_i[:, 0], dphi_j[:, 0]) + np.multiply(dphi_i[:, 1], dphi_j[:, 1]), alpha) * det_j
    )

    return ke


# --- Load vector
def load_u(sig_np1k: np.ndarray, b: np.ndarray, phi_i: np.ndarray, dphi_i: np.ndarray, alpha: np.array, det_j: float):
    # initialize load vector
    lu = np.zeros(2, dtype=np.float64)

    # calculate load vector
    for igp in range(0, alpha.shape[0]):
        lu[0] = (
            lu[0]
            + phi_i[igp] * b[igp, 0] * alpha[igp] * det_j
            - dphi_i[igp, 0] * sig_np1k[igp, 0] * alpha[igp] * det_j
            - dphi_i[igp, 1] * sig_np1k[igp, 2] * alpha[igp] * det_j
        )
        lu[1] = (
            lu[1]
            + phi_i[igp] * b[igp, 1] * alpha[igp] * det_j
            - dphi_i[igp, 1] * sig_np1k[igp, 1] * alpha[igp] * det_j
            - dphi_i[igp, 0] * sig_np1k[igp, 2] * alpha[igp] * det_j
        )

    return lu


def load_t(
    matpar: list,
    theta_0: float,
    dt: float,
    DthetaDt_t_dt: np.ndarray,
    DthetaDx: np.ndarray,
    trDepsDt_t_dt: np.ndarray,
    rhor: np.ndarray,
    phi_i: np.ndarray,
    dphi_i: np.ndarray,
    alpha: np.array,
    det_j: float,
):
    # Extract material parameters
    lhS = matpar[0]
    mhS = matpar[1]
    alpha_t = matpar[2]
    rho_0 = matpar[3]
    c_t = matpar[4]
    k_t = matpar[5] * dt

    # initialize load vector
    lt = 0.0

    # calculate load vector
    beta = alpha_t * (2 * mhS + 3 * lhS)

    for igp in range(0, alpha.shape[0]):
        l_l = phi_i[igp] * dt * rhor[igp]
        l_res = (
            phi_i[igp] * rho_0 * c_t * DthetaDt_t_dt[igp]
            + phi_i[igp] * beta * theta_0 * trDepsDt_t_dt[igp]
            + k_t * (dphi_i[igp, 0] * DthetaDx[igp, 0, 0] + dphi_i[igp, 1] * DthetaDx[igp, 0, 1])
        )

        lt = lt + (l_l - l_res) * alpha[igp] * det_j

    return lt


# --- Assembly on the element level ---
def assemble_system_elmt(c: int, kernel_data: KernelData, dt: float, matvec: list, theta_0: float):
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
    dphi_x = np.einsum("mk,mij->kij", jinv, dphi)

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
    DuDx = interpolate_grad(dofs_u, dphi_x, bs=2)
    divu = DuDx[:, 0, 0] + DuDx[:, 1, 1]

    # evaluate divergence of velocity (time n+1, iterate k) at quadrature points
    DDuDtDx_t_dt = interpolate_grad(dofs_u - dofs_u_n, dphi_x, bs=2)
    trDepsDt_t_dt = DDuDtDx_t_dt[:, 0, 0] + DDuDtDx_t_dt[:, 1, 1]

    # evaluate temperature (time n+1, iterate k) at quadrature points
    theta = interpolate(dofs_t, phi, bs=1)[:, 0]

    # evaluate time-derivative of temperature (time n+1, iterate k) at quadrature points
    DthetaDt_t_dt = interpolate(dofs_t - dofs_t_n, phi, bs=1)[:, 0]

    # evaluate the temperature gradient (time n+1, iterate k) at quadrature points
    DthetaDx = interpolate_grad(dofs_t, dphi_x, bs=1)

    # evaluate stress (time n+1, iterate k) at quadrature points (Voigt notation!)
    lhS = matvec[0]
    mhS = matvec[1]
    alpha_t = matvec[2]
    beta = alpha_t * (2 * mhS + 3 * lhS)

    sigma = np.zeros((alpha.shape[0], 3))
    sigma[:, 0] = 2 * mhS * DuDx[:, 0, 0] + lhS * divu[:] - beta * (theta[:] - theta_0)
    sigma[:, 1] = 2 * mhS * DuDx[:, 1, 1] + lhS * divu[:] - beta * (theta[:] - theta_0)
    sigma[:, 2] = mhS * (DuDx[:, 0, 1] + DuDx[:, 1, 0])

    # assemble stiffness and load vector per element
    for ii in range(0, nodes_per_elmt):
        # calculate local dof indices for displacement/ temperature DOFs
        iu1 = kernel_data.get_node_dofs_local(ii, 0)[0]
        iu2 = iu1 + 1
        it = kernel_data.get_node_dofs_local(ii, 1)

        # evaluate load vector contributions
        l_u = load_u(sigma, vec_b, phi[0, :, ii, 0], dphi_x[:, :, ii].T, alpha, det_j)
        l_t = load_t(
            matvec,
            theta_0,
            dt,
            DthetaDt_t_dt,
            DthetaDx,
            trDepsDt_t_dt,
            vec_rhor,
            phi[0, :, ii, 0],
            dphi_x[:, :, ii].T,
            alpha,
            det_j,
        )

        # fill the values into the elemental load vector
        le[iu1] = l_u[0]
        le[iu2] = l_u[1]
        le[it] = l_t

        for jj in range(0, nodes_per_elmt):
            # calculate local dof indices for first and second DOF
            ju1 = kernel_data.get_node_dofs_local(jj, 0)[0]
            ju2 = ju1 + 1
            jt = kernel_data.get_node_dofs_local(jj, 1)

            # evaluate local stiffness constributions
            k_uu = stiffness_element_uu(matvec, dphi_x[:, :, ii].T, dphi_x[:, :, jj].T, alpha, det_j)
            k_ut = stiffness_element_ut(matvec, dphi_x[:, :, ii].T, phi[0, :, jj, 0], alpha, det_j)
            k_tu = stiffness_element_tu(matvec, theta_0, phi[0, :, ii, 0], dphi_x[:, :, jj].T, alpha, det_j)
            k_tt = stiffness_element_tt(
                matvec, dt, phi[0, :, ii, 0], phi[0, :, jj, 0], dphi_x[:, :, ii].T, dphi_x[:, :, jj].T, alpha, det_j
            )

            # fill the values into the elemental stiffness matrix
            # stiffness due to displacement
            ke[iu1, ju1] = k_uu[0, 0]
            ke[iu1, ju2] = k_uu[0, 1]
            ke[iu2, ju1] = k_uu[1, 0]
            ke[iu2, ju2] = k_uu[1, 1]

            # stiffness displacment/ temperature coupling
            ke[iu1, jt] = k_ut[0]
            ke[iu2, jt] = k_ut[1]

            # stiffness temperature/ displacement coupling
            ke[it, ju1] = k_tu[0]
            ke[it, ju2] = k_tu[1]

            # stiffness due to temperature
            ke[it, jt] = k_tt

    return ke, le


# --- Assembly on the global level ---
def assemble_system(kernel_data: KernelData, dt: float, matvec: list, theta_0: float):
    # counters
    number_of_cells = kernel_data.get_ncells_mesh()
    ndofs_global = kernel_data.get_ndofs_glob()

    # initialize tangents
    A = sp.lil_matrix((ndofs_global, ndofs_global))
    L = np.zeros(ndofs_global, dtype=np.float64)

    # loop over all elements
    for c in range(0, number_of_cells):
        # calculate elemental stiffness
        Ke, Le = assemble_system_elmt(c, kernel_data, dt, matvec, theta_0)

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
