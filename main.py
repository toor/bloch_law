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
    
    # Compute the nearest neighbours for all defined symmetry directions.
    print("Computing nns in all bases")
    nearest_neighbours = lattice.in_plane_nearest_neighbours(3, plot_image=True, prefix=prefix)
    
    for s, nns in nearest_neighbours.items():
        #print(f"Symmetry direction: {s}")
        data_dir = prefix + f"/nn_{lattice_type}/{s}/"
        #print(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        filename = data_dir + "nn_ip.dat"
        
        (in_plane, plane_above, plane_below) = nns
        #print(f"{in_plane.shape[0]}, {out_of_plane.shape[0]}")
        #print(type(in_plane))
        #print(type(out_of_plane))
        
        # print("Computing in-plane nearest neighbours in original basis.\n")
        with open(filename, "w") as f:
            f.write('idx    n1    n2    n3\n')
           
            for i, nn in enumerate(in_plane):
                #print(f"nn shape: {nn.shape}")
                coeffs = lattice.nearest_neighbours_original_basis(s, nn, "Z=0")
                
                #print("Has Bravais lattice coefficients (in conventional basis):\n")
                #lattice.print_vector(coeffs, True)

                f.write(f'{i}    {coeffs[0]}    {coeffs[1]}    {coeffs[2]}\n')

            f.close()

        filename = data_dir + "nn_oop_above.dat"
        
        #print("Computing nearest neighbours in plane above in original basis\n")
        with open(filename, "w") as f:
            f.write('idx    n1    n2    n3\n')

            for i, nn in enumerate(plane_above):
                coeffs = lattice.nearest_neighbours_original_basis(s, nn, "Z=+d")
                #print("Has Bravais lattice coefficients (in conventional basis):\n")
                #lattice.print_vector(coeffs, True)
                #print(f"int of -1 is {int(-1.0)}")

                f.write(f'{i}    {coeffs[0]}    {coeffs[1]}    {coeffs[2]}\n')

            f.close()

        filename = data_dir + "nn_oop_below.dat"
        
        #print("Computing nearest neighbours in plane below in original basis\n")
        with open(filename, "w") as f:
            f.write('idx    n1    n2    n3\n')

            for i, nn in enumerate(plane_below):
                coeffs = lattice.nearest_neighbours_original_basis(s, nn, "Z=-d")
                #print("Has Bravais lattice coefficients (in conventional basis):\n")
                #lattice.print_vector(coeffs, True)
                #print(f"int of -1 is {int(-1.0)}")

                f.write(f'{i}    {coeffs[0]}    {coeffs[1]}    {coeffs[2]}\n')

            f.close()
#
#
#        test_1 = np.array([0, 0.5, 0.5])
#        test_2 = np.array([0, -0.5, -0.5])
#        A = np.array([[-1., 1., 1.],
#                     [1.,-1.,1.],
#                     [1.,1.,-1.]], dtype=np.float64)
#        print(A @ test_1)
#        print(A @ test_2)
