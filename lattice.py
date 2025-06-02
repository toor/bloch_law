import numpy as np
import os
import matplotlib.pyplot as plt

ATOL = 1e-6
np.set_printoptions(threshold=np.inf)

def print_r_text(input):
    print('\x1b[1;31;40m' + input + '\x1b[0m')


class Params:
    def __init__(self, params):
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
    def unpack(self):
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

class Basis:
    def __init__(self, lattice_type, params):
        (a, b, c, alpha, beta, gamma) = params.unpack()
        match lattice_type: 
            case "cP":
                self.basis = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
            case "cI":
                self.basis = (a/2)*np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64) 
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
            case "cF":
                self.basis = (a/2)*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0],[1,1,0], [1,1,1]], dtype=np.float64)
                # case "tP":
                #     self.basis = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]], dtype=np.float64)
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0]], dtype=np.float64)
                # case "tI":
                #     self.basis = np.array([[a, -a, c], [a, a, c], [-a, -a, c]], dtype=np.float64) / 2
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
                # case "oP":
                #     self.basis = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64)
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [0,1,1]], dtype=np.float64)
                # case "oS":
                #     self.basis = np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [0, 0, c]], dtype=np.float64)
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64) 
                # case "oI":
                #     self.basis = np.array([[a, b, c], [-a, b, c], [a, -b, c]], dtype=np.float64) / 2
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
                # case "oF":
                #     self.basis = np.array([[0, b, c], [a, 0, c], [a, b, 0]], dtype=np.float64) / 2
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64)
                # case "hP":
                #     self.basis = np.array([[a, 0, 0], [-a/2, a*np.sqrt(3)/2, 0], [0, 0, c]], dtype=np.float64)
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0]], dtype=np.float64)
                # case "hR":
                #     self.basis = self.__gen_rhombohedral(a, alpha)
                #     self.sym_dirs = np.array([[1,1,1], [1,1,0]], dtype=np.float64)
                # case "mP":
                #     self.basis = np.array([[a, 0, 0], [0, b, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64)
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0]], dtype=np.float64)
                # case "mS":
                #     self.basis = np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64)
                #     self.sym_dirs = np.array([[1,0,0], [0,1,0]], dtype=np.float64)
                # case "aP":
                #     self.basis = self.__gen_triclinic(a, b, c, alpha, beta, gamma)
                #     self.sym_dirs = None
    
    def __gen_rhombohedral(self, a, alpha):
        a1 = [1,0,0]
        a2 = [np.cos(alpha), np.sin(alpha), 0]
        a3 = [np.cos(alpha),
              (1 - np.cos(alpha))*np.cos(alpha)/np.sin(alpha),
              np.sqrt(1 - 3*np.cos(alpha)**2 + 2*np.cos(alpha)**3)/np.sin(alpha)]
        return a*np.array([a1, a2, a3])

    def __gen_triclinic(self, a, b, c, alpha, beta, gamma): 
        return np.array([[a, 0, 0],
                    [b*np.cos(gamma), b*np.sin(gamma), 0],
                    [c*np.cos(beta),
                     c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma),
                     c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma))**2)]], dtype=np.float64)



class Lattice:
    def __init__(self, lattice_type, params):
        basis = Basis(lattice_type, params)
        try:
            num_bases = basis.sym_dirs.shape[0]
        except AttributeError as e:
            print(f"Caught an AttributeError {e} in Lattice() with lattice type {lattice_type}")
        #self.bases = np.zeros((num_bases, 3, 3))
        self.original_basis = basis.basis
        self.params = params 

        bases = {}
        for i, sym_dir in enumerate(basis.sym_dirs):
            print(sym_dir)
            rotated_basis = self.rotate_vectors(basis.basis, sym_dir)
            dict_i = {self.sym_dir_to_string(sym_dir): rotated_basis}
            bases.update(dict_i)

        self.bases = bases
        self.ltype = lattice_type
    
    def new_lattice(self, sym_dir, reps):
        basis = self.bases[sym_dir]
        a1 = basis[0,:]
        a2 = basis[1,:]
        a3 = basis[2,:]
        
        points = []

        for n1 in range(-reps, reps+1):
            for n2 in range(-reps, reps+1):
                for n3 in range(-reps, reps+1):
                    points.append(n1*a1 + n2*a2 + n3*a3)
        #print(np.array(points))
        return np.array(points)

    def select_atoms_by_plane(self, lattice, z):
        plane = []
        for p in lattice:
            if np.abs(p[2] - z) < ATOL:
                r = np.linalg.norm(p)
                #print(f"Found coordinates of atom in plane z={z} with distance from origin {r}")
                plane.append(p)
        return np.array(plane)

    def sort_by_distance(self, points, x, y, z):
        return np.array(sorted(points,
            key=lambda p: np.sqrt((p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2)))

    def return_vectors_by_length(self, vectors, length):
        points = []
        for vec in vectors:
            if np.isclose(np.linalg.norm(vec), length):
                points.append(vec)
        return np.array(points)
               
    def sym_dir_to_string(self, sym_dir):
        sym_dir = sym_dir.astype(np.int64)

        return f'{sym_dir[0]}{sym_dir[1]}{sym_dir[2]}'

    def string_to_sym_dir(self, string):
        sym_dir = []
        for c in string:
            sym_dir.append(float(int(c)))
        return np.array(sym_dir, dtype=np.float64)

    # Note to self: retur
    def calculate_interplanar_spacing(self, s):
        chars = [c for c in s]
        h = int(chars[0])
        k = int(chars[1])
        l = int(chars[2])

        lat_fam = [c for c in self.ltype][0]
        match lat_fam:
            case "c":
                return self.params.a / np.sqrt(h**2 + k**2 + l**2)
            case "t":
                return 1 / np.sqrt((h**2 + k**2)/(self.params.a**2) + (self.params.c**2)/(l**2))
            case "o":
                return 1 / np.sqrt((h/self.params.a)**2 + (k/self.params.b)**2 + (l/self.params.c)**2)
            case "h":
                return 1 / np.sqrt(4*(h**2 + h*k + k**2)/(3*(self.params.a**2)) + (l/self.params.c)**2)

    def rotate_vectors(self, vectors, final_axis):
        R = self.rot_matrix(final_axis)
        
        new_vectors = []
        for i, _ in enumerate(vectors):
            vec = vectors[i,:]
            new_vectors.append(R @ vec)
        return np.array(new_vectors)
    
    def rot_matrix(self, final_axis, inverse=False):
        # Extract the miller indices, and compute the rotation axis using 
        # the miller indices as coefficients of the originab lattice basis vectors. 
        a = final_axis.copy().astype(np.float64)
        b = self.original_basis
        uprime = a[0]*b[0,:] + a[1]*b[1,:] + a[2]*b[2,:]
        uprime /= np.linalg.norm(uprime)
        print(f"Rotation axis (x,y,z) = ({uprime[0]},{uprime[1]}, {uprime[2]})")

        u = final_axis.copy().astype(np.float64)
        u /= np.linalg.norm(u)

        v = np.array([0,0,1])

        # Compute angle of rotation
        dot_prod = np.dot(u, v)
        theta = np.arccos(dot_prod)
       
        # Compute axis of rotation
        r = np.cross(u, v)
        r /= np.linalg.norm(r) 

        r_shear = np.array([[0, -r[2], r[1]],
                            [r[2], 0, -r[0]],
                            [-r[1], r[0], 0]])
        r_shear2 = r_shear @ r_shear

        R = np.eye(3) + np.sin(theta)*r_shear + (1- np.cos(theta))*r_shear2
        #print(f'Rotation matrix to align with symmetry direction {self.sym_dir_to_string(final_axis)} is {R}')
        if inverse:
            return np.linalg.inv(R)
        return R 

    def plot_lattice(self, points, filename, vectors=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw the lattice
        ax.scatter(points[:,0], points[:,1], points[:,2], c='gold', s=10)
        ax.set_xlim(np.min(points[:,0]), np.max(points[:,0]))
        ax.set_ylim(np.min(points[:,1]), np.max(points[:,1]))
        #ax.set_zlim(np.min(points[:,2]), np.max(points[:,2]))
        # Draw some atomic planes which intersect the lattice at multiples of the [111]
        # interplanar spacing, d = a/sqrt(3)
        if vectors is not None:
            for vec in vectors:
                ax.quiver(0.0, 0.0, 0.0, vec[0], vec[1], vec[2], color='red', arrow_length_ratio=0.2)
    
        ax.set_xticks(ticks=[0.0, 1.0, -1.0], labels=['x=0', '1', '-1'])
        ax.set_yticks(ticks=[0.0, 1.0, -1.0], labels=['y=0', '1', '-1'])
        ax.set_zticks(ticks=[0.0, 1.0, -1.0], labels=['z=0', '1', '-1'])
        plt.show()

        fig.savefig(filename)
        plt.close()

    def in_plane_nearest_neighbours(self, reps, plot_image=False, prefix=None):
        if prefix is not None:
            figs_dir = prefix + f'/figures/{self.ltype}/'
            os.makedirs(figs_dir, exist_ok=True)
        nn = {}
        for s, b in self.bases.items():
            print(f'lattice type {self.ltype} with sym. dir. {s}\n')
            points = self.new_lattice(s, reps)
            sym_dir = self.string_to_sym_dir(s)
            planar_spacing = self.calculate_interplanar_spacing(s) 

            # Save an image of the lattice before we perform any other calculations on it 
            filename = figs_dir + f"base_lattice_{s}.png"
            #self.plot_lattice(points, filename)

            #print(points)
            # Select all atoms which lie in the plane z = 0, and sort these 
            # by distance to the origin.
            x = y = z = 0.0
            # Remove first element; this will be the origin itself.
            z0_plane = self.sort_by_distance(np.unique(
                self.select_atoms_by_plane(points, z), axis=0), x,y,z)
            zplus_plane = self.sort_by_distance(np.unique(
                self.select_atoms_by_plane(points, planar_spacing), axis=0), x,y,z)
            zminus_plane = self.sort_by_distance(np.unique(
                self.select_atoms_by_plane(points, -planar_spacing), axis=0), x,y,z)
        
            print(f'PLANE Z=0 shape = {z0_plane.shape}\n\n')
            #for p in z0_plane:
            #    print(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}); r={np.linalg.norm(p):.4f}")
            #print("\n")
            print(f'PLANE Z=+d shape = {zplus_plane.shape}\n\n')
            
            #for p in zplus_plane:
            #   print(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}); r={np.linalg.norm(p):.4f}")
            #print("\n")
            
            print(f'PLANE Z=-d shape = {zminus_plane.shape}')
            #for p in zminus_plane:
            #    print(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}); r={np.linalg.norm(p):.4f}")
            #print('\n')
            
            shortest_length_inplane = np.linalg.norm(z0_plane[1,:])
            shortest_length_above = np.linalg.norm(zplus_plane[0,:])
            shortest_length_below = np.linalg.norm(zminus_plane[0,:])

            #print(f'points in plane z = 0: {z0_plane}')
            # shortest_inplane_vector = z0_plane[1,:]
            # min_length = np.linalg.norm(shortest_vector)
            
            three_planes = np.row_stack((z0_plane[1:],
                                         self.return_vectors_by_length(zplus_plane,
                                                                       shortest_length_above),
                                         self.return_vectors_by_length(zminus_plane,
                                                                       shortest_length_below)))
            # Select only those vectors which match the required length.
            nn_vectors = np.row_stack((self.return_vectors_by_length(z0_plane,
                                                                     shortest_length_inplane),
                                       self.return_vectors_by_length(zplus_plane,
                                                                     shortest_length_above),
                                       self.return_vectors_by_length(zminus_plane,
                                                                     shortest_length_below)))
            nn_count = nn_vectors.shape[0]

            out_of_plane_neighbours = np.row_stack((self.return_vectors_by_length(zplus_plane,
                                                                     shortest_length_above),
                                       self.return_vectors_by_length(zminus_plane,
                                                                     shortest_length_below)))
            oo_nn_count = out_of_plane_neighbours.shape[0]
            print(f"Located {oo_nn_count} neighbours out of the plane.")
            in_plane_neighbours = self.return_vectors_by_length(z0_plane, shortest_length_inplane)
            ip_nn_count = in_plane_neighbours.shape[0]
            print(f"Located {ip_nn_count} neighbours in the plane")

            neighbours = (in_plane_neighbours, out_of_plane_neighbours)
            
            if plot_image:
                filename = figs_dir + f"nn_plot_{s}.png"
                self.plot_lattice(three_planes, filename, vectors=nn_vectors)

            #print(f"Got {nn_count} nearest neighbours for lattice type {self.ltype} in symmetry direction {s}\n with distance {min_length} from origin.")
            nn.update({s: neighbours})
        
        return nn
    
    def print_vector(self, vector, use_colour=False):
        if use_colour:
            print_r_text(f"(x,y,z) = ({vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f})\n")
        else:
            print(f"(x,y,z) = ({vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f})\n")
    

    # For every nn_vector v_i, we have v_i = M (n1, n2, n3)^T, 
    # where n1, n2 and n3 are integer coefficients of the basis vectors, 
    # and M = (a1, a2, a3). This function allows us to find the nearest neighbour 
    # vectors in the original un-rotated basis.
    def nearest_neighbours_original_basis(self, sym_dir, vector):
        #print(f"Computing integer coefficients of nearest neighbours for lattice {self.ltype} with symmetry direction {sym_dir}")
        basis = self.original_basis 
        a1 = basis[0,:]#.reshape((3,1))
        a2 = basis[1,:]#.reshape((3,1))
        a3 = basis[2,:]#.reshape((3,1))

        A = np.column_stack((a1,a2,a3))
        #print(A)
        R = self.rot_matrix(self.string_to_sym_dir(sym_dir), inverse=True)
        print(f'R shape = {R.shape}')

        A_inv = np.linalg.inv(A)
        print("Matrix to compute basis coefficients is: \n")
        print(A_inv)
        #print("Vector to be rotated:\n")
        self.print_vector(vector)
        #print("Rotation of vector:\n")
        self.print_vector(R @ vector, True)
        rot_vector = R @ vector
        print(f'rotated vector shape = {rot_vector.shape}')
        # first return the vector to the original basis, then compute it 
        # in terms of integer coefficients of the original basis vectors.
        v =  A_inv @ rot_vector 
        return v.astype(np.int64)


