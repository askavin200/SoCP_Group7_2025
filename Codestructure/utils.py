# --- Imports ---
import numpy as np
import typing

import ufl
import dolfinx.fem as dfem


# --- Interpolate functions ---
def interpolate(u: np.ndarray, phi: np.ndarray, bs=2) -> np.ndarray:
    # Reshape u
    u_r = u.reshape((-1, bs))
    phi_r = phi[0, :, :, 0]
    return np.dot(phi_r, u_r)


# --- Interpolate gradients ---
def interpolate_grad(u: np.ndarray, dphi: np.ndarray, bs=2) -> np.ndarray:
    u_r = u.reshape((-1, bs))
    return np.einsum("ij, kli -> ljk", u_r, dphi, optimize="optimal")


# Derivative of C_inv with respect to C
def dCinvdC(C_inv: np.ndarray):

    dCinvdC = -0.5 * (
        np.einsum("ik, jl -> ijkl", C_inv, C_inv)
        + np.einsum("il, jk -> ijkl", C_inv, C_inv)
    )

    return dCinvdC


# --- Cinv_dyad_Cinv in Voigt notation ---
def Cinv_dyad_Cinv(C):
    C_inv = np.linalg.inv(C)
    dim = C_inv.shape[0]
    Cinv_dyad_Cinv = np.zeros((dim, dim, dim, dim))
    Cinv_dyad_Cinv = 0.5 * np.einsum("ij,kl->ijkl", C_inv, C_inv)
    return Cinv_dyad_Cinv


def convert_4_order_voigt(A):
    # Convert 4th order tensor to Voigt notation
    A_Voigt = np.zeros((3, 3))
    A_Voigt[0, 0] = A[0, 0, 0, 0]
    A_Voigt[1, 1] = A[1, 1, 1, 1]
    A_Voigt[2, 2] = A[0, 1, 0, 1]
    A_Voigt[0, 1] = A[0, 0, 1, 1]
    A_Voigt[1, 0] = A[1, 1, 0, 0]
    A_Voigt[0, 2] = A[0, 0, 0, 1]
    A_Voigt[2, 0] = A[0, 1, 0, 0]
    A_Voigt[1, 2] = A[1, 1, 0, 1]
    A_Voigt[2, 1] = A[0, 1, 1, 1]
    return A_Voigt


def convert_2_order_voigt(A):
    # Convert 2nd order tensor to Voigt notation
    A_voigt = np.zeros((3))
    A_voigt[0] = A[0, 0]
    A_voigt[1] = A[1, 1]
    A_voigt[2] = 2 * A[0, 1]
    return A_voigt


# Returns the Second Piola Kirchoff Stress in Voigt notation
def convert_S_voigt(S):
    # Convert 2nd order tensor to Voigt notation
    S_voigt = np.zeros((3))
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[0, 1]
    return S_voigt


# Returns the discretized gradient operator
def B_nonlinear(F: np.ndarray, dphi_i: np.ndarray):
    B_nl = np.zeros((3, 2))
    B_nl[0, 0] = F[0, 0] * dphi_i[0]
    B_nl[0, 1] = F[1, 0] * dphi_i[0]
    B_nl[1, 0] = F[0, 1] * dphi_i[1]
    B_nl[1, 1] = F[1, 1] * dphi_i[1]
    B_nl[2, 0] = F[0, 0] * dphi_i[1] + F[0, 1] * dphi_i[0]
    B_nl[2, 1] = F[1, 0] * dphi_i[1] + F[1, 1] * dphi_i[0]
    return B_nl


# --- Project stresses ---
class ProjectStress:
    def __init__(
        self,
        V_ut: dfem.FunctionSpace,
        matpar_lambda: typing.Union[float, dfem.Function],
        matpar_mu: typing.Union[float, dfem.Function],
        matpar_beta: typing.Union[float, dfem.Function],
        theta_0: float,
        f_stress: typing.Callable[
            [
                dfem.Function,
                dfem.Function,
                typing.Union[float, dfem.Function],
                typing.Union[float, dfem.Function],
                typing.Union[float, dfem.Function],
                float,
            ],
            typing.Any,
        ],
    ):
        # Storage for projected stress
        self.V_stress = dfem.FunctionSpace(V_ut.mesh, ("DG", 0))
        self.stress = dfem.Function(self.V_stress)
        self.stress.name = "vMises_stress"

        # Storage for uh and thetah
        self.uh_u = dfem.Function(V_ut.sub(0).collapse()[0])
        self.uh_theta = dfem.Function(V_ut.sub(1).collapse()[0])

        # The stress expression
        stress_ufl = f_stress(
            self.uh_u, self.uh_theta, matpar_lambda, matpar_mu, matpar_beta, theta_0
        )
        stress_dev_ufl = stress_ufl - 0.5 * ufl.tr(stress_ufl) * ufl.Identity(2)
        stress_vM = ufl.sqrt(1.5 * ufl.inner(stress_dev_ufl, stress_dev_ufl))

        self.stress_expr = dfem.Expression(
            stress_vM, self.V_stress.element.interpolation_points()
        )

    def evaluate_stress(self, uh: dfem.Function):
        # Update storage
        self.uh_u.x.array[:] = uh.sub(0).collapse().x.array[:]
        self.uh_theta.x.array[:] = uh.sub(1).collapse().x.array[:]

        # Interpolate von Mises stress
        self.stress.interpolate(self.stress_expr)

    def copy_stress(self):
        # New function
        stress = dfem.Function(self.V_stress)
        stress.x.array[:] = self.stress.x.array[:]

        return stress
