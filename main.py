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

lattice_types = ["cP", "cI", "cF"]

for lattice_type in lattice_types:
    print(f"Lattice with type {lattice_type}")
    p = lat.Params(params)
    lattice = lat.Lattice(lattice_type, p)
    
    # repeat the lattice three times along each basis vector.
    nearest_neighbours = lattice.in_plane_nearest_neighbours(3, plot_image=True, prefix=prefix)
    
    for s, nns in nearest_neighbours.items():
        print(f"Symmetry direction: {s}")
        data_dir = prefix + f"/nn_{lattice_type}/{s}/"
        #print(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        filename = data_dir + f"{s}_nn_inplane.dat"
        
        (in_plane, out_of_plane) = nns
        print(f"{in_plane.shape[0]}, {out_of_plane.shape[0]}")
        #print(type(in_plane))
        #print(type(out_of_plane))
        
        print("Computing in-plane nearest neighbours in original basis.\n")
        with open(filename, "w") as f:
            f.write('idx    n1    n2    n3\n')
           
            for i, nn in enumerate(in_plane):
                coeffs = lattice.nearest_neighbours_original_basis(s, nn)
                
                #print("Has Bravais lattice coefficients (in conventional basis):\n")
                lattice.print_vector(coeffs, True)

                f.write(f'{i}    {coeffs[0]}    {coeffs[1]}    {coeffs[2]}\n')

            f.close()

        filename = data_dir + f"{s}_nn_outofplane.dat"
        
        print("Computing out-of-plane nearest neighbours in original basis\n")
        with open(filename, "w") as f:
            f.write('idx    n1    n2    n3\n')

            for i, nn in enumerate(out_of_plane):
                coeffs = lattice.nearest_neighbours_original_basis(s, nn)
                #print("Has Bravais lattice coefficients (in conventional basis):\n")
                lattice.print_vector(coeffs, True)

                f.write(f'{i}    {coeffs[0]}    {coeffs[1]}    {coeffs[2]}\n')

            f.close()

        test_1 = np.array([0, 0.5, 0.5])
        test_2 = np.array([0, -0.5, -0.5])
        A = np.array([[-1, 1, 1],
                     [1,-1,1],
                     [1,1,-1]], dtype=np.float64)
        print(A @ test_1)
        print(A @ test_2)
