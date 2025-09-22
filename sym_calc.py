import sympy as sp 
import numpy as np 

coeffs = sp.Matrix([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
in_plane = sp.Matrix([[0,0,1],[0,0,-1]])
above = sp.Matrix([[1,0,0],[0,1,0]])
below = sp.Matrix([[-1,0,0],[0,-1,0]])

def arr_f64(arr):
    return np.array(arr, dtype=np.float64)

def sym_arr(arr):
    return sp.Matrix(arr)



coeff_dict = {
    "sc": {
        "001": {
            "above": sym_arr([0,0,1]),
            "below": sym_arr([0,0,-1]),
            "inplane": sym_arr([[1,0,0],[-1,0,0],[0,1,0],[0,1,0]])
        },
        "111": {
            "above": sym_arr([[1,0,0],[0,1,0],[0,0,1]]),
            "below": sym_arr([[-1,0,0],[0,-1,0],[0,0,-1]])
            "inplane": None
        },
        "110": {
            "above": sym_arr([[1,0,0],[0,1,0]]),
            "below": sym_arr([[-1,0,0],[0,-1,0]]),
            "inplane": sym_arr([[0,0,1],[0,0,-1]])
        }
    },
    "bcc": {
        "001": {
            "above": sym_arr([[-1,1,1],[1,1,1],[-1,-1,1],[1,-1,1]]),
            "below": sym_arr([[-1,1,-1],[1,1,-1],[-1,-1,-1],[1,-1,-1]]),
            "inplane": None
        },
        "111": {
            "above": sym_arr([[-1,1,1],[1,-1,1],[1,1,-1]]),
            "below": sym_arr([[-1,-1,1],[-1,1,-1],[1,-1,-1]]),
            "inplane": None
        },
        "110": {
            "above": sym_arr([[1,1,1],[1,1,-1]]),
            "below": sym_arr([[-1,-1,1],[-1,-1,-1]]),
            "inplane": sym_arr([[-1,1,1],[1,-1,1],[-1,1,-1],[1,-1,-1]])
        }
    },
    "fcc": {
        "001": {
            "above": sym_arr([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]])
            "below": sym_arr([[1,0,-1],[-1,0,-1],[-1,0,-1],[0,1,-1]])
            "inplane": sym_arr([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])
        },
        "111": {
            "above": sym_arr([[1,1,0],[1,0,1],[0,1,1]]),
            "below": sym_arr([[-1,-1,0],[-1,0,-1],[0,-1,-1]]),
            "inplane": sym_arr([[-1,1,0],[1,-1,0],[-1,0,1],[0,-1,1],[1,0,-1],[0,1,-1]])
        },
        "110": {
            "above": sym_arr([[1,0,1],[0,1,1],[1,0,-1],[0,1,-1]]),
            "below": sym_arr([[-1,0,1],[0,-1,1],[-1,0,-1],[0,-1,-1]]),
            "inplane": sym_arr([[-1,1,0],[1,-1,0]])
        }
    }
}

class DispersionRelation:
    def __init__(self, coeffs_a, coeffs_b, coeffs_i, coeff_map, name):
        self.coeffs_a = sp.Matrix([c for c in coeffs_a])
        self.coeffs_b = sp.Matrix([c for c in coeffs_b])
        self.coeffs_i = sp.Matrix([c for c in coeffs_i])

        self.a = sp.symbols('a')
        self.coeff_map
    
    def delta_110(self, coeffs):
        root = sp.sqrt(2)
        return [coeffs[0] - coeffs[1] - coeffs[2]*root2,
                -coeffs[0] + coeffs[1] - coeffs[2]*root2]/2
    
    def delta_111(self, coeffs):
        mu_1, mu_1, mu_3 = sp.symbols('mu_1 mu_2 mu_3')

        return [mu_1*coeffs[0] + mu_2*coeffs[1] - mu_3*coeffs[2],
                mu_2*coeffs[0] + mu_1*coeffs[1] + mu_3*coeffs[2]]

    def compute_nn_001(self):
        if self.name == "fcc" or self.name == "bcc":
            pref = self.a / 2 
        else:
            pref = self.a

        q_x, q_y = sp.symbols('q_x q_y')
        
        s_n = sp.symbols('s_n')
        s_nplus = sp.symbols('s_{n+1}')
        s_nminus = sp.symbols('s_{n-1}')
        
        z = 0
        if self.name = "fcc":
            z = 12 
        elif self.name = "bcc":
            z = 8
        else:
            z = 6
        
        sum = z*s_n 
        for c_a in self.coeffs_a:
            cprime_a = self.coeff_map(c_a)

            delta = pref*Matrix([cprime_a[0], cprime_a[1]])
            sum += s_nplus*sp.exp(sp.I*delta.dot(q))

        for c_b in self.coeffs_b:
            cprime_b = self.coeff_map(c_b)

            delta = pref*Matrix([cprime_b[0], cprime_b[1]])
            sum -= s_nminus*sp.exp(sp.I*delta.dot(q))

        for c_i in self.coeffs_i:
            cprime_i = self.coeff_map(c_i)

            delta = pref*Matrix([cprime_i[0], cprime_i[1]])
            sum -= s_n*sp.exp(sp.I*delta.dot(q))

        return sum 

    def compute_nn_110(self):
        if self.name == "fcc" or self.name == "bcc":
            pref = self.a/2 
        else:
            pref = self.a 

       
        q_x, q_y = sp.symbols('q_x q_y')
        q = Matrix([q_x, q_y])

        s_n = sp.symbols('s_n')
        s_nplus = sp.symbols('s_{n+1}')
        s_nminus = sp.symbols('s_{n-1}')
        
        z = 0 
        if self.name == "fcc":
            z = 10 
        elif self.name == "bcc":
            z = 8
        else: 
            z = 6
        sum = z*s_n 

        for c_a in self.coeffs_a:
            cprime_a = self.coeff_map(c_a)
             
            delta = pref*Matrix(self.delta_110(cprime_a))
            sum -= s_nplus*sp.exp(sp.I*delta.dot(q))
        for c_b in self.coeffs_b:
            cprime_b = self.coeff_map(c_b)

            delta = pref*Matrix(self.delta_110(cprime_b))
            sum -= s_nminus*sp.exp(sp.I*delta.dot(q))
        for c_i in self.coeffs_i:
            cprime_i = self.coeff_map(c_i)

            delta = pref*Matrix(self.delta_110(cprime_i))
            sum -= s_n*sp.exp(sp.I*delta.dot(q))

        return sum

    def compute_nn_111(self):
        if self.name == "fcc" or self.name == "bcc":
            pref = self.a/2 
        else:
            pref = self.a 
        
       
        q_x, q_y = sp.symbols('q_x q_y')
        q = Matrix([q_x, q_y])


        s_n = sp.symbols('s_n')
        s_nplus = sp.symbols('s_{n+1}')
        s_nminus = sp.symbols('s_{n-1}')
         
        
        z = 0 
        if self.name == "fcc":
            z = 12
        else: 
            z = 6 

        sum = z*s_n

        for c_a in self.coeffs_a:
            cprime_a = self.coeff_map(c_a)
             
            delta = pref*Matrix(self.delta_111(cprime_a))
            sum -= s_nplus*sp.exp(sp.I*delta.dot(q))
        for c_b in self.coeffs_b:
            cprime_b = self.coeff_map(c_b)

            delta = pref*Matrix(self.delta_111(cprime_b))
            sum -= s_nminus*sp.exp(sp.I*delta.dot(q))
        for c_i in self.coeffs_i:
            cprime_i = self.coeff_map(c_i)

            delta = pref*Matrix(self.delta_111(cprime_i))
            sum -= s_n*sp.exp(sp.I*delta.dot(q))

        return sum

omega_fcc = DispersionRelation()

s_0, s_a, s_b = sp.symbols('s_0 s_a s_b')

a = sp.symbols('a')
root2 = sp.sqrt(2)

sum = 6*s_0

for i in range(2):
    a_coeff = above.row(i)
    b_coeff = below.row(i)
    i_coeff = in_plane.row(i)


    delta_a = a*sp.Matrix([a_coeff[0] - a_coeff[1] - a_coeff[2]*root2,
                    -a_coeff[0] + a_coeff[1] - a_coeff[2]*root2])/2

    delta_b = a*sp.Matrix([b_coeff[0] - b_coeff[1] - b_coeff[2]*root2,
                    -b_coeff[0] + b_coeff[1] - b_coeff[2]*root2])/2

    delta_i = a*sp.Matrix([i_coeff[0] - i_coeff[1] - i_coeff[2]*root2,
                    -i_coeff[0] + i_coeff[1] - i_coeff[2]*root2])/2

    dot_a = s_a*sp.exp(sp.I*delta_a.dot(q))
    dot_b = s_b*sp.exp(sp.I*delta_b.dot(q))
    dot_i = s_0*sp.exp(sp.I*delta_i.dot(q))

    sum -= (dot_a + dot_b + dot_i)


sum = sp.trigsimp(sum)
sum = sum.rewrite(sp.cos)
sum = sp.collect(sum, [s_0, s_a, s_b])


with open('test.tex', "w") as f:
    f.write(sp.latex(sum))
