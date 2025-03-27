import numpy as np

ATOL = 1e-6
np.set_printoptions(threshold=np.inf)


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
            case "tP":
                self.basis = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0]], dtype=np.float64)
            case "tI":
                self.basis = np.array([[a, -a, c], [a, a, c], [-a, -a, c]], dtype=np.float64) / 2
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
            case "oP":
                self.basis = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [0,1,1]], dtype=np.float64)
            case "oS":
                self.basis = np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64) 
            case "oI":
                self.basis = np.array([[a, b, c], [-a, b, c], [a, -b, c]], dtype=np.float64) / 2
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
            case "oF":
                self.basis = np.array([[0, b, c], [a, 0, c], [a, b, 0]], dtype=np.float64) / 2
                self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64)
            case "hP":
                self.basis = np.array([[a, 0, 0], [-a/2, a*np.sqrt(3)/2, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0]], dtype=np.float64)
            case "hR":
                self.basis = self.__gen_rhombohedral(a, alpha)
                self.sym_dirs = np.array([[1,1,1], [1,1,0]], dtype=np.float64)
            case "mP":
                self.basis = np.array([[a, 0, 0], [0, b, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0]], dtype=np.float64)
            case "mC":
                self.basis = np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0]], dtype=np.float64)
            case "aP":
                self.basis = self.__gen_triclinic(a, b, c, alpha, beta, gamma)
                self.sym_dirs = None

    def __gen_rhombohedral(self, a, alpha):
        return a*np.array([[1, 0, 0],
                       [np.cos(alpha), np.sin(alpha), 0],
                       [np.cos(alpha), (1-np.cos(alpha))*np.cos(alpha)/np.sin(alpha), np.sqrt((1-np.cos(alpha))(1 + 2*np.cos(alpha))/(1 + np.cos(alpha)))]], dtype=np.float64) 

    def __gen_triclinic(self, a, b, c, alpha, beta, gamma): 
        return np.array([[a, 0, 0],
                    [b*np.cos(gamma), b*np.sin(gamma), 0],
                    [c*np.cos(beta),
                     c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma),
                     c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma))**2)]], dtype=np.float64)



class Lattice:
    def __init__(self, lattice_type, params):
        basis = Basis(lattice_type, params)
        num_bases = basis.sym_dirs.shape[0]
        #self.bases = np.zeros((num_bases, 3, 3))
        self.original_basis = basis.basis

        bases = {}
        for i, sym_dir in enumerate(basis.sym_dirs):
            rotated_basis = self.rotate_vectors(basis.basis, sym_dir)
            dict_i = {self.sym_dir_to_string(sym_dir): rotated_basis}
            bases.update(dict_i)
        self.bases = bases
    
    def new_lattice(self, sym_dir, reps):
        basis = self.bases[sym_dir]
        a1 = basis[0,:]
        a2 = basis[1,:]
        a3 = basis[2,:]

        points = []
        for n1 in range(-reps, reps+1):
            for n2 in range(-reps, reps+1):
                for n3 in range(-reps, reps+1):
                    vec = n1*a2 + n2*a2 + n3*a3
                    points.append(vec)
        return np.array(points)

    def select_atoms_by_plane(self, lattice, z):
        plane = []
        for p in lattice:
            if np.abs(p[2] - z) < ATOL:
                plane.append(p)
        return np.array(p)

    def sort_by_distance(self, points, x, y, z):
        return np.array(sorted(points, key=lambda p: np.sqrt((p[0] - x)**2 + (p[1] - y)**2)))

    def return_vectors_by_length(self, vectors, length):
        points = []
        for vec in vectors:
            if np.isclose(np.linalg.norm(vec), length):
                points.append(vec)
        return np.array(points)
               
    def sym_dir_to_string(self, sym_dir):
        sym_dir = sym_dir.astype(np.int64)

        return f'{sym_dir[0]}{sym_dir[1]}{sym_dir[2]}'

    def rotate_vectors(self, vectors, final_axis):
        R = self.rot_matrix(final_axis)

        for vec in vectors:
            new_vec = R @ vec 
            vec = new_vec
        return vectors
    
    def rot_matrix(self, final_axis):
        #print(f'Rotating {miller_indices.astype(np.int64)} to align with axis {rot_axis.astype(np.int64)}')
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

        return R 

    def in_plane_nearest_neighbours(self, reps):
        nn = {}
        for s, b in self.bases.items():
            print(s)
            print(b)
            points = self.new_lattice(s, reps)
            print(points)
            #points = points.reshape((points.shape[0], 3, 1))
            # Select all atoms which lie in the plane z = 0, and sort these 
            # by distance to the origin.
            x = y = z = 0.0
            # Remove first element; this will be the origin itself.
            z0_plane = self.select_atoms_by_plane(points, z)
            print(f'z0_plane has shape {z0_plane.shape}')
            z0_plane = self.sort_by_distance(z0_plane, x, y, z)
            #print(f'z0_plane has shape {z0_plane.shape}')
            shortest_vector = z0_plane[0,:]
            print(shortest_vector.shape)
            min_length = np.linalg.norm(z0_plane[0,:])
            

            nn_vectors = self.return_vectors_by_length(z0_plane, min_length)
            nn.update({s: nn_vectors})
            
        return nn
    
    # For every nn_vector v_i, we have v_i = M (n1, n2, n3)^T, 
    # where n1, n2 and n3 are integer coefficients of the basis vectors, 
    # and M = (a1, a2, a3). This function allows us to find the nearest neighbour 
    # vectors in the original un-rotated basis.
    def nearest_neighbours_original_basis(self, vector):
        basis = self.original_basis 
        a1 = basis[0,:].reshape((3,1))
        a2 = basis[1,:].reshape((3,1))
        a3 = basis[2,:].reshape((3,1))

        M = np.column_stack((a1,a2,a3))
        print(M.shape)
        print(vector.shape)
        m = np.linalg.inv(M)

        v =  m @ vector 
        return v 


