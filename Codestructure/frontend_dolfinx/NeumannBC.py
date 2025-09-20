# --- Imports ---
import numpy as np
from petsc4py import PETSc
import typing

import ufl

import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc

# --- NeumannBC ---
class NeumannBC:
    def __init__(self, 
                 V_ut: dfem.FunctionSpace, 
                 facet_function: typing.Any, 
                 time_function: typing.Optional[typing.Callable] = None) -> None:
        # The surface integrator
        self.ds = ufl.Measure("ds", domain=V_ut.mesh, subdomain_data=facet_function)
        
        # The test functions
        vu, vt = ufl.TestFunctions(V_ut)
        self.test_functions = [vu, vt]
        
        # Physical time as dfem-constant
        self.time_ufl = dfem.Constant(V_ut.mesh, float(0))
        
        # Transient conditions
        self.time_function = time_function
        self.value_time_function = dfem.Constant(V_ut.mesh, float(1.0))
        
        # Initialise load vector
        self.vector_initialised = False
        self.rhs_bc = 0
        
        self.form_rhs_bc = None
        self.L = None
    
    def set_traction(self, 
                     traction_vector: typing.List[float], 
                     sub_surfaces: typing.List) -> None:
        # The surface integrator 
        ds_cur = self.ds(sub_surfaces[0])
        for i in range(1, len(sub_surfaces)):
            ds_cur += self.ds(sub_surfaces[i])
        
        #  Append RHS
        self.rhs_bc += self.value_time_function * ufl.inner(ufl.as_vector(traction_vector), self.test_functions[0]) * ds_cur
    
    def set_heatflux(self, 
                     heatflux: float, 
                     sub_surfaces: typing.List) -> None:
        # The surface integrator 
        ds_cur = self.ds(sub_surfaces[0])
        for i in range(1, len(sub_surfaces)):
            ds_cur += self.ds(sub_surfaces[i])
        
        #  Append RHS
        self.rhs_bc -= self.value_time_function * heatflux * self.test_functions[1] * ds_cur
            
    def evaluate_neumann_bc(self, time) -> np.ndarray:
        if not self.vector_initialised:
            # Initialise load vector
            self.form_rhs_bc = dfem.form(self.rhs_bc)
            self.L = dfem_petsc.create_vector(self.form_rhs_bc)
            
            self.vector_initialised = True
            
        # Update time
        self.time_ufl.value = time
        
        # Reset value of time-function
        if self.time_function is not None:
            self.value_time_function.value = self.time_function(time)
            
        # Reassemble load vector
        with self.L.localForm() as loc:
            loc.set(0)
            
        dfem.petsc.assemble_vector(self.L, self.form_rhs_bc)
        self.L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        return self.L.array