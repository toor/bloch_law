class Params:
    def __init(self, params):
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
class Basis:
    def __init__(type):
        basis = 
        match type: 
            case "cP":
                 (np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=np.float64),
                        np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64))
            case "cI":
                 ((a/2)*np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64), 
                        np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64))
            case "cF":
                 ((a/2)*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64), np.array([[1,0,0], [0,1,0],[1,1,0], [1,1,1]], dtype=np.float64))
            case "tP":
                return (np.array([[a, 0, 0], [0, a, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0],  [1,1,0]], dtype=np.float64))
            case "tI":
                return (np.array([[a, -a, c], [a, a, c], [-a, -a, c]], dtype=np.float64) / 2, np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64))
            case "oP":
                return (np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [1,1,0], [0,1,1]], dtype=np.float64))
            case "oS":
                return (np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64)) 
            case "oI":
                return (np.array([[a, b, c], [-a, b, c], [a, -b, c]], dtype=np.float64) / 2, np.array([[1,0,0], [0,1,0], [1,1,0], [1,1,1]], dtype=np.float64))
            case "oF":
                return (np.array([[0, b, c], [a, 0, c], [a, b, 0]], dtype=np.float64) / 2, np.array([[1,0,0], [0,1,0],  [1,1,0], [0,1,1]], dtype=np.float64))
            case "hP":
                return (np.array([[a, 0, 0], [-a/2, a*np.sqrt(3)/2, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [1,1,0]], dtype=np.float64))
            case "hR":
                return (gen_rhombohedral(a, alpha), np.array([[1,1,1], [1,1,0]], dtype=np.float64))
            case "mP":
                return (np.array([[a, 0, 0], [0, b, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64), np.array([[1,0,0], [0,1,0]], dtype=np.float64))
            case "mC":
                return (np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64), np.array([[1,0,0], [0,1,0]]], dtype=np.float64))
            case "aP":
            return (gen_triclinic(a, b, c, alpha, beta, gamma), None)



class Lattice:
    def __init__(self, params, type):
        self.params = Params(params) 

        


