import numpy as np

import basix
import dolfinx.fem as dfem

from .handle_basix import fspace_to_basix, create_quadrature_rule


# --- Kernel data (element integration kernel) ---
class KernelData:
    def __init__(self, fspace: dfem.FunctionSpace) -> None:
        # Store function space
        self.V = fspace

        # Set solution function/ history field
        self.U = dfem.Function(self.V)
        self.Un = dfem.Function(self.V)

        # The volumetric loads
        self.load_blm = None
        self.load_boe = None

        # --- Counters
        # Spacial dimension
        self.gdim = fspace.mesh.geometry.dim

        # Number of DOFs
        self.ndofs = fspace._cpp_object.dofmap.index_map.size_global

        # Number of DOFs per cell
        self.ndof_cell = None

        # DOFs displacement per cell
        self.ndof_cell_u = None
        self.bs_u = self.V.sub(0).collapse()[0]._cpp_object.dofmap.index_map_bs

        # DOFs temperature per cell
        self.ndof_cell_t = None

        # DOFs geometry per cell
        self.ndof_cell_geom = None

        # --- Extract relevant data
        # Mesh
        self.msh = fspace.mesh

        # DOFmap
        self.dofmap = fspace.dofmap.list
        self.dofmap_geom = fspace.mesh.geometry.dofmap

        # Shape functions
        self.shpfkt = None
        self.shpfkt_geom = None
        self.dphi_geom = None

        # Quadrature rule
        self.q_points = None
        self.q_alpha = None

        # --- Identifires
        self.qrule_is_set = False
        self.shpfkt_is_set = False
        self.isomap_is_set = False

    # --- Tabulate shape functions
    def tabulate_shpfkt(self, calc_first_derivatives=True) -> None:
        if self.qrule_is_set is False:  
            raise RuntimeError("Quadrature rule not set")

        # Set identifire for initialisation
        self.shpfkt_is_set = True

        # Create basix element
        belmt = fspace_to_basix(self.V.sub(0).collapse()[0])

        # Tabulate shape functions
        if calc_first_derivatives:
            self.shpfkt = belmt.tabulate(1, self.q_points)
        else:
            self.shpfkt = belmt.tabulate(0, self.q_points)
        
        # Set number of DOFs per cell
        dim_fspace = belmt.dim
        self.ndof_cell_u = self.bs_u * dim_fspace
        self.ndof_cell_t = dim_fspace
        self.ndof_cell = self.ndof_cell_u + self.ndof_cell_t

    def tabulate_shpfkt_geom(self) -> None:
        # Create basix element for geometry
        bfamily = basix.finite_element.string_to_family("Lagrange", self.msh.topology.cell_type.name)
        bcelltype = basix.cell.string_to_type(self.msh.topology.cell_type.name)   
        c_element = basix.create_element(bfamily, bcelltype, 1, basix.LagrangeVariant.gll_warped)

        # Tabulate shape functions
        self.shpfkt_geom = c_element.tabulate(1, np.array([[0, 0]]))

        # Get mumbers of DOFs per cell
        self.ndof_cell_geom = c_element.dim

    # --- Isoparametric concept
    def init_isomap(self) -> None:
        # Set idetifire for initialisation
        self.isomap_is_set = True

        # Tabulate shape-functions
        self.tabulate_shpfkt_geom()

        # Extract derivatives of shape functions
        self.dphi_geom = self.shpfkt_geom[1:self.gdim + 1, 0, :, 0].copy()

    def isomap_elmt(self, c: int):
        if self.isomap_is_set is False:
            raise RuntimeError("Isoparametric concept not initialised")
        
        # Extract geometry data
        geometry = np.zeros((self.ndof_cell_geom, self.gdim), dtype=np.float64)
        geometry[:] = self.msh.geometry.x[self.dofmap_geom.links(c), :self.gdim]

        # Calculate jacobi matrix
        J_q = np.dot(geometry.T, self.dphi_geom.T)

        return J_q, np.abs(np.linalg.det(J_q))

    # --- Setter functions
    def set_quadrature_rule(self, order: int) -> None:
        # Set identifire for initialisation
        self.qrule_is_set = True

        # Set quadrature rule
        self.q_points, self.q_alpha = create_quadrature_rule(self.msh.ufl_cell().cellname(), order)

    def set_volumeforces(self, fe_b) -> None:
        self.load_blm = fe_b

    def set_heatsource(self, fe_r) -> None:
        self.load_boe = fe_r

    # --- Getter functions
    # Counters (geometry)
    def get_ncells_mesh(self):
        return self.msh.topology.index_map(self.gdim).size_global

    def get_nnodes_mesh(self):
        return self.msh.topology.index_map(0).size_global
    
    # Counter (finite element)
    def get_nnodes_per_cell(self):
        return self.ndof_cell_t
    
    def get_ndofs_glob(self):
        return self.ndofs

    def get_ndofs_per_cell(self):     
        return self.ndof_cell

    def get_ndofs_temp_per_cell(self):
        return self.ndof_cell_t
    
    def get_ndofs_disp_per_cell(self):
        return self.ndof_cell_u
    
    # Shape functions/ quadrature
    def get_number_quadrature_points(self):
        return self.q_points.shape[0]

    def get_quadrature_points(self):
        return self.q_points
    
    def get_quadrature_weights(self):
        return self.q_alpha

    def get_shape_functions(self):
        if self.shpfkt_is_set is False:
            raise RuntimeError("Shape functions not tabulated")

        return self.shpfkt
    
    # DOFs
    def get_cell_dofs(self, c: int, subspace=None):
        # Cell DOFs
        cdofs = self.dofmap.links(c)

        if subspace is None:
            return cdofs
        else:
            begin = (self.ndof_cell_u, 0)[subspace == 0]
            end = (self.ndof_cell, self.ndof_cell_u)[subspace == 0]
            return cdofs[begin:end]

    def get_node_dofs(self, c: int, n: int, subspace=None):
        cdofs = self.get_cell_dofs(c, subspace)

        if subspace is None:
            ndofs = np.zeros(self.bs_u + 1, dtype=np.int32)
            ndofs[0:self.bs_u] = cdofs[n * self.bs_u:(n + 1) * self.bs_u]
            ndofs[self.bs_u] = cdofs[self.ndof_cell_u + n]

            return ndofs
        elif subspace == 0:
            return cdofs[n * self.bs_u:(n + 1) * self.bs_u]
        elif subspace == 1:
            return cdofs[n]  
        else:
            raise RuntimeError("No such subspace!")

    def get_node_dofs_local(self, n: int, subspace=None):
        if subspace is None:
            ndofs = np.zeros(self.bs_u + 1, dtype=np.int32)
            ndofs[0:self.bs_u] = range(n * self.bs_u, (n + 1) * self.bs_u)
            ndofs[self.bs_u] = self.ndof_cell_u + n

            return ndofs
        elif subspace == 0:
            ndofs = np.zeros(self.bs_u , dtype=np.int32)
            ndofs[0:self.bs_u] = range(n * self.bs_u, (n + 1) * self.bs_u)
            return ndofs
        elif subspace == 1:
            return self.ndof_cell_u + n  
        else:
            raise RuntimeError("No such subspace!")

    # Solution u_np1^k resp. u_n 
    def get_cell_solution_np1(self, c: int, subspace = None):
        # Get cell DOFs
        dofs = self.get_cell_dofs(c, subspace)

        return self.U.x.array[dofs]
    
    def get_cell_solution_n(self, c: int, subspace = None):
        # Get cell DOFs
        dofs = self.get_cell_dofs(c, subspace)

        return self.Un.x.array[dofs]

    # Volumetric loads
    def get_volume_forces(self, c: int, subspace : int):
        if subspace == 0:
            if self.load_blm is None:
                return None
            else:
                # extract DOFs
                h = self.load_blm.function_space.dofmap.list.links(c)
                
                # condider block size
                dofs = np.repeat(h * 2, self.bs_u)
                dofs[1::2] += 1
                
                return self.load_blm.x.array[dofs]
        elif subspace == 1:
            if self.load_boe is None:
                return None
            else:
                dofs = self.load_boe.function_space.dofmap.list.links(c)
                return self.load_boe.x.array[dofs]
        else:
            raise RuntimeError("No such subspace!")


