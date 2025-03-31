import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import lattice as lat
import os

params = {
    "a": 1.0,
    "b": 1.5,
    "c": 2.0,
    "alpha": np.pi/4,
    "beta": np.pi/3,
    "gamma": np.pi/6
}

prefix = os.getcwd()
print(prefix)


lattice_types = ["cP", "cI", "cF", "tP", "tI", "oP", "oS", "oI", "oF", "hP", "mP", "mS"] #"hR", "mP", "mS"]
#lattice_types = ["cF"]
#lattice_types = ["hR"]

for lattice_type in lattice_types:
    #print(lattice_type)
    p = lat.Params(params)
    lattice = lat.Lattice(lattice_type, p)
    
    # repeat the lattice three times along each basis vector.
    nearest_neighbours = lattice.in_plane_nearest_neighbours(3, plot_image=True, prefix=prefix)
    
    for s, nns in nearest_neighbours.items():
        data_dir = prefix + f"/nn_{lattice_type}/{s}/"
        #print(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        filename = data_dir + f"{s}__nn.dat"

        with open(filename, "w") as f:
            f.write('idx    n1    n2    n3\n')
            for i, nn in enumerate(nns):
                coeffs = lattice.nearest_neighbours_original_basis(s, nn)
                f.write(f'{i}    {coeffs[0]}    {coeffs[1]}    {coeffs[2]}\n')
            f.close()
        
