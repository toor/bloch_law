class Params:
    def __init(self, params):
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
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64))
            case "cI":
                self.basis = (a/2)*np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64) 
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
            case "cF":
                self.basis = (a/2)*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0],[1,1,0], [1,1,1]], dtype=np.float64))
            case "tP":
                self.basis = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0]], dtype=np.float64)
            case "tI":
                self.basis = np.array([[a, -a, c], [a, a, c], [-a, -a, c]], dtype=np.float64) / 2
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64)
            case "oP":
                self.basis = (np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [0,1,1]], dtype=np.float64)
            case "oS":
                self.basis = np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [0, 0, c]], dtype=np.float64)
                self.sym_dirs = np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64) 
            case "oI":
                self.basis = np.array([[a, b, c], [-a, b, c], [a, -b, c]], dtype=np.float64) / 2
                self.sym_dirs = np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64))
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
                self.sym_dirs = np.array([[1,0,0], [0,1,0]]], dtype=np.float64)
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
        
        bases = []
        for i, sym_dir in enumerate(basis.sym_dirs):
            rotated_basis = self.rotate_vectors(basis.basis, sym_dir)
            bases.append((self.sym_dir_to_string(sym_dir), rotated_basis))
        bases = np.array(bases)
        self.bases = bases


    def search_for_basis(self, sym_dir):
        for basis in self.bases:
            (s,b) = basis 
            # Compare label for basis
            if s == sym_dir:
                return b

               
    def sym_dir_to_string(sym_dir):
        sym_dir = sym_dir.astype(np.int64)

        return f'{sym_dir[0]}{sym_dir[1]}{sym_dir[2]}'


    
    def rot_matrix(final_axis):
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


    def select_atoms_by_plane(self, plane):
        for vec in self.vectors


