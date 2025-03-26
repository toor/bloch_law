import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import lattice as lat

params = {
    "a": 1.0,
    "b": 1.5,
    "c": 2.0,
    "alpha": np.pi/4,
    "beta": np.pi/3,
    "gamma": np.pi/6
}

lattice_types = ["cP", "cI", "cF", "tP", "tI", "oP", "oS", "oI", "oF", "hP", "hR", "mP", "mS"]

for lattice_type in lattice_types:
    lattice = Lattice(lattice_type, params)

        
