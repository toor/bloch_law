import numpy as np 
from sympy import *
import matplotlib.pyplot as plt 

half = Rational(1,2)
sqrt3 = sqrt(3)

def arr_f64(arr):
    return np.array(arr, dtype=np.float64)
kpar = Matrix([k_x, k_y])


# We basically want to evaluate the sum over nearest neighbours in:
# Case 1) The bulk. Must sum over in-plane nearest neighbours and out-of-plane.
# Case 2) The bottom surface (n=1). Must sum over in-plane neighbours and those in the plane above (n=2).
# Case 3) The top surface (n=N). Must sum over in-plane neighbours and those in the plane below (n = N - 1).

class Basis:
    def __init__(self, lattice_type):
        match lattice_type:
            case "cP":
                self.basis = Matrix([[a,0,0],[0,a,0],[0,0,a]])
                self.sym_dir = ["100", "010", "110", "111"]
            case "cI":
                self.basis = a*half*Matrix([[-1,1,1],[1,-1,1],[1,1,-1]])
                self.sym_dir = ["100", "010", "110", "111"]
            case "cF":
                self.basis = a*half*Matrix([[0,1,1],[1,0,1],[1,1,0]])
                self.sym_dir = ["100", "010", "110", "111"]
