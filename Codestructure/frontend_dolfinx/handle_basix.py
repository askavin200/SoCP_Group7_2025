import numpy as np
from typing import Tuple

import basix

import dolfinx.fem as dfem

# --- Handle basix-element ---
def fspace_to_basix(fspace: dfem.FunctionSpace):
    # Extract mesh
    msh = fspace.mesh
    
    # Extract cell type
    ct = basix.cell.string_to_type(msh.topology.cell_type.name)
    
    # Extract element family
    family = fspace.ufl_element().family()
    if family == "Q":
        family = "Lagrange"
        
    bfamily = basix.finite_element.string_to_family(family, msh.topology.cell_type.name)
        
    return basix.create_element(bfamily, ct, fspace.ufl_element().degree(), basix.LagrangeVariant.gll_warped)

# --- Tabulate shape functions ---
def tabulate_basix_elmt(points, family, degree, cell_type) -> np.ndarray:
    # Creta basix element
    elmt = basix.create_element(family, cell_type, degree, basix.LagrangeVariant.gll_warped)

    return elmt.tabulate(1, points)

# --- Create quadrature rule ---
def create_quadrature_rule(cell_name, order) -> Tuple[np.ndarray, np.ndarray]:
    basix_cell = basix.cell.string_to_type(cell_name)
    q_points, q_alpha = basix.make_quadrature(basix_cell, order)

    return q_points, q_alpha
