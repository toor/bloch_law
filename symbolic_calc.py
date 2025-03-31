import numpy as np 
from sympy import *
import matplotlib.pyplot as plt 

half = Rational(1,2)
sqrt3 = sqrt(3)

def arr_f64(arr):
    return np.array(arr, dtype=np.float64)

class Basis:
    def __init__(self, lattice_type):
        match lattice_type:
            case "cP":
                self.basis = Matrix([[a,0,0],[0,a,0],[0,0,a]])
                self.sym_dir = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "cI":
                self.basis = a*half*Matrix([[-1,1,1],[1,-1,1],[1,1,-1]])
                self.sym_dir = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "cF":
                self.basis = a*half*Matrix([[0,1,1],[1,0,1],[1,1,0]])
                self.sym_dir = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "tP":
                self.basis = Matrix([[a,0,0],[0,a,0],[0,0,c]])
                self.sym_dir = arr_f64([[1,0,0],[0,1,0],[1,1,0]])
            case "tI":
                self.basis = Matrix([[a,-a,c],[a,a,c],[-a,-a,c]])
                self.sym_dir = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "oP":
                self.basis = Matrix([[a,0,0],[0,b,0],[0,0,c]])
                self.sym_dirs = arr_f64([[1,0,0,],[0,1,0],[1,1,0],[1,1,1]])
            case "oS":
                self.basis = Matrix([[a*half, b*half, 0], [-a*half, b*half, 0], [0,0,c]])
                self.sym_dirs = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "oI":
                self.basis = Matrix([[a,b,c],[-a,b,c],[a,-b,c]])
                self.sym_dirs = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "oF":
                self.basis = Matrix([[0,b,c],[a,0,c],[a,b,0]])
                self.sym_dirs = arr_f64([[1,0,0],[0,1,0],[1,1,0],[1,1,1]])
            case "hP":
                self.basis = Matrix([[a,0,0],[-half*a, a*sqrt3*half, 0], [0,0,c]])
                self.sym_dirs = arr_f64([[1,0,0],[0,1,0],[1,1,0]])
            case "hR":
                self.basis =  

