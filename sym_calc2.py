import numpy as np 
import sympy as sp 

def sym_arr(arr):
    return sp.Matrix(arr)

mu_1, mu_2, mu_3 = sp.symbols('mu_1 mu_2 mu_3', real=True, positive=True)
root2 = sp.sqrt(2)
q = sym_arr(sp.symbols('q_x q_y', real=True, positive=True))
j = sp.I 
a = sp.symbols('a', real=True, positive=True)

s_n, sa, sb = sp.symbols('s_n, s_{n+1}, s_{n-1}')

def nn_110(coeffs, const):
    m1 = coeffs[0]
    m2 = coeffs[1]
    m3 = coeffs[2]
    return const*sym_arr([m1 - m2 - m3*root2,
                          -m1 + m2 - m3*root2]) / 2 

def nn_111(coeffs, const):
    m1 = coeffs[0]
    m2 = coeffs[1]
    m3 = coeffs[2]

    return const*sym_arr([mu_1*m1 + mu_2*m2 - mu_3*m3,
                          mu_2*m1 + mu_1*m2 + mu_3*m3])

def nn_001(coeffs, const):
    m1 = coeffs[0]
    m2 = coeffs[1]
    return const*sym_arr([m1, m2])

def nn_map(coeffs, prefactor, sym_dir):
    match sym_dir:
        case "001":
            return nn_001(coeffs, prefactor)
        case "110":
            return nn_110(coeffs, prefactor)
        case "111":
            return nn_111(coeffs, prefactor)


coeff_dict = {
    "sc": {
        "001": {
            "above": sym_arr([[0,0,1]]),
            "below": sym_arr([[0,0,-1]]),
            "inplane": sym_arr([[1,0,0],[-1,0,0],[0,1,0],[0,1,0]]),
            "count": 6
        },
        "111": {
            "above": sym_arr([[1,0,0],[0,1,0],[0,0,1]]),
            "below": sym_arr([[-1,0,0],[0,-1,0],[0,0,-1]]),
            "inplane": None,
            "count": 6
        },
        "110": {
            "above": sym_arr([[1,0,0],[0,1,0]]),
            "below": sym_arr([[-1,0,0],[0,-1,0]]),
            "inplane": sym_arr([[0,0,1],[0,0,-1]]),
            "count": 6
        },
        "prefactor": a
    },
    "bcc": {
        "001": {
            "above": sym_arr([[-1,1,1],[1,1,1],[-1,-1,1],[1,-1,1]]),
            "below": sym_arr([[-1,1,-1],[1,1,-1],[-1,-1,-1],[1,-1,-1]]),
            "inplane": None,
            "count": 8
        },
        "111": {
            "above": sym_arr([[-1,1,1],[1,-1,1],[1,1,-1]]),
            "below": sym_arr([[-1,-1,1],[-1,1,-1],[1,-1,-1]]),
            "inplane": None,
            "count": 6
        },
        "110": {
            "above": sym_arr([[1,1,1],[1,1,-1]]),
            "below": sym_arr([[-1,-1,1],[-1,-1,-1]]),
            "inplane": sym_arr([[-1,1,1],[1,-1,1],[-1,1,-1],[1,-1,-1]]),
            "count": 8
        },
        "prefactor": a/2
    },
    "fcc": {
        "001": {
            "above": sym_arr([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]]),
            "below": sym_arr([[1,0,-1],[-1,0,-1],[0,-1,-1],[0,1,-1]]),
            "inplane": sym_arr([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]]),
            "count": 12
        },
        "111": {
            "above": sym_arr([[1,1,0],[1,0,1],[0,1,1]]),
            "below": sym_arr([[-1,-1,0],[-1,0,-1],[0,-1,-1]]),
            "inplane": sym_arr([[-1,1,0],[1,-1,0],[-1,0,1],[0,-1,1],[1,0,-1],[0,1,-1]]),
            "count": 12
        },
        "110": {
            "above": sym_arr([[1,0,1],[0,1,1],[1,0,-1],[0,1,-1]]),
            "below": sym_arr([[-1,0,1],[0,-1,1],[-1,0,-1],[0,-1,-1]]),
            "inplane": sym_arr([[-1,1,0],[1,-1,0]]),
            "count": 10
        },
        "prefactor": a/2
    }
}

lattice_types = ["sc", "bcc", "fcc"]
sym_dirs = ["001", "110", "111"]

for lat_type in lattice_types:
    print(f"LATTICE TYPE: {lat_type}\n")
    pref = coeff_dict[lat_type]["prefactor"]
    for sym_dir in sym_dirs:
        print(f"SYMMETRY DIRECTION: {sym_dir}")
        coeffs_a = coeff_dict[lat_type][sym_dir]["above"]
        coeffs_b = coeff_dict[lat_type][sym_dir]["below"]
        coeffs_i = coeff_dict[lat_type][sym_dir]["inplane"]
        num_a = sp.shape(coeffs_a)[0]
        num_b = sp.shape(coeffs_b)[0]
        num_i = sp.shape(coeffs_i)[0] if coeffs_i is not None else 0

        print(f"neighbours above: {num_a}")
        print(f"neighbours below: {num_b}")
        print(f"neighbours inplane: {num_i}")

        c = coeff_dict[lat_type][sym_dir]["count"]
        sum = c*s_n
        
        for i in range(num_a):
            ca = coeffs_a.row(i)
            delta = nn_map(ca, pref, sym_dir)
            sum -= sa*sp.exp(j*q.dot(delta))
        
        for i in range(num_b):
            cb = coeffs_b.row(i)
            delta = nn_map(cb, pref, sym_dir)
            sum -= sb*sp.exp(j*q.dot(delta))
        
        if coeffs_i is not None:
            for i in range(num_i):
                ci = coeffs_i.row(i)
                delta = nn_map(ci, pref, sym_dir)
                sum -= s_n*sp.exp(j*q.dot(delta))
        
        sum = sp.trigsimp(sum)
        sum = sum.rewrite(sp.cos)
        sum = sp.collect(sum, [s_n, sa, sb])

        filename = lat_type + "_" + sym_dir + ".tex"
        with open(filename, "w") as f:
            f.write(sp.latex(sum))
