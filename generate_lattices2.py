import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def get_bravais_lattices(a, b, c, theta, beta, gamma, alpha): 
    lattices = {
        "sc": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([0, a, 0]),
            "a3": np.array([0, 0, a])
        },
        "bcc": {
            "a1": (a/2) * np.array([-1,  1,  1]),
            "a2": (a/2) * np.array([ 1, -1,  1]),
            "a3": (a/2) * np.array([ 1,  1, -1])
        },
        "fcc": {
            "a1": (a/2) * np.array([0,  1,  1]),
            "a2": (a/2) * np.array([1,  0,  1]),
            "a3": (a/2) * np.array([1,  1,  0])
        },
        "tetr_sc": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([0, a, 0]),
            "a3": np.array([0, 0, c])
        },
        "tetr_bcc": {
            "a1": 0.5 * np.array([ a, -a,  c]),
            "a2": 0.5 * np.array([ a,  a,  c]),
            "a3": 0.5 * np.array([-a, -a,  c])
        },
        "ortho_sc": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([0, b, 0]),
            "a3": np.array([0, 0, c])
        },
        "ortho_base": {
            "a1": 0.5 * np.array([ a,  b, 0]),
            "a2": 0.5 * np.array([-a,  b, 0]),
            "a3": np.array([0, 0, c])
        },
        "ortho_bcc": {
            "a1": 0.5 * np.array([ a,  b,  c]),
            "a2": 0.5 * np.array([-a,  b,  c]),
            "a3": 0.5 * np.array([ a, -b,  c])
        },
        "ortho_fcc": {
            "a1": 0.5 * np.array([0,  b,  c]),
            "a2": 0.5 * np.array([a,  0,  c]),
            "a3": 0.5 * np.array([a,  b, 0])
        },
        "hex": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([-a/2, (a*np.sqrt(3))/2, 0]),
            "a3": np.array([0, 0, c])
        },
        "trig": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([a*np.cos(theta), a*np.sin(theta), 0]),
            "a3": np.array([
                a*np.cos(theta),
                a*(np.cos(theta) - np.cos(theta)**2 + np.sin(theta)**2) / np.sin(theta),
                a*np.sqrt(1 - 3*np.cos(theta)**2 + 2*np.cos(theta)**3) / np.sin(theta)
            ])
        },
        "mono_sc": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([0, b, 0]),
            "a3": np.array([c*np.cos(beta), 0, c*np.sin(beta)])
        },
        "mono_base": {
            "a1": 0.5 * np.array([a, b, 0]),
            "a2": 0.5 * np.array([-a, b, 0]),
            "a3": np.array([c*np.cos(beta), 0, c*np.sin(beta)])
        },
        "tric": {
            "a1": np.array([a, 0, 0]),
            "a2": np.array([b*np.cos(gamma), b*np.sin(gamma), 0]),
            "a3": np.array([
                c*np.cos(beta),
                c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma),
                c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma))**2)
            ])
        }
    }
    return lattices

def generate_high_symmetry_directions(a, c=None):
    if c is None:
        c = a  # Assume c = a if not provided
    
    cells = {
        # Cubic
        "sc_100": np.array([1,0,0]),
        "bcc_100": np.array([1,0,0]),
        "bcc_110": np.array([1,1,0]),
        "bcc_111": np.array([1,1,1]),
        "fcc_100": np.array([1,0,0]),
        "fcc_110": np.array([1,1,0]),
        "fcc_111": np.array([1,1,1]),
        # Simple tetragonal
        "st_100": np.array([1,0,0]),
        "st_110": np.array([1,1,0]), 
        "st_001": np.array([0,0,1]),
        # Body-centred tetragonal
        "bct_100": np.array([1,0,0]),
        "bct_010": np.array([0,1,0]),
        "bct_001": np.array([0,0,1]),
        # Simple orthorhombic 
        "so_100": np.array([1,0,0]),
        "so_010": np.array([0,1,0]),
        "so_001": np.array([0,0,1]),
        # Base-centred orthorhombic is the same as simple orthorhombic
        "baco_100": np.array([1,0,0]),
        "baco_010": np.array([0,1,0]),
        "baco_001": np.array([0,0,1]),
        # Body-centred orthorhombic 
        "bco_100": np.array([1,0,0]),
        "bco_010": np.array([0,1,0]),
        "bco_001": np.array([0,0,1]),
        "bco_110": np.array([1,1,0]),
        "bco_111": np.array([1,1,1])
        # Hexagonal
        "sh_100": np.array([1,0,0]),
        "sh_010": np.array([0,1,0])
        "sh_110": np.array([1,1,0]),
        "sh_001": np.array([0,0,1]),
        # Rhombohedral
        "rh_111": np.array([1,1,1]),
        "rh_110": np.array([1,1,0])
        # Monoclinic 
        "mc_100": np.array([1,0,0]),
        "mc_001": np.array([0,0,1])
    }
   
    return cells

def plot_plane(fig, ax, plane_z):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y)
    Z = np.full_like(X, plane_z)
    
    ax.plot_surface(X, Y, Z, color='red', alpha=0.1)

def plot_lattice(points, a, plot_planes):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    d_plane = a / np.sqrt(3)

    # Draw the lattice
    ax.scatter(points[:,0], points[:,1], points[:,2], c='gold', s=10)
    ax.set_xlim(np.min(points[:,0]), np.max(points[:,0]))
    ax.set_ylim(np.min(points[:,1]), np.max(points[:,1]))
    #ax.set_zlim(np.min(points[:,2]), np.max(points[:,2]))
    # Draw some atomic planes which intersect the lattice at multiples of the [111]
    # interplanar spacing, d = a/sqrt(3)
    if plot_planes == True:
        for i in range(5):
            plane_z = i*d_plane 
            plot_plane(fig, ax, plane_z)

    ax.set_xticks(ticks=[0.0], labels=['x=0'])
    ax.set_yticks(ticks=[0.0], labels=['y=0'])
    ax.set_zticks(ticks=[0.0], labels=['z=0'])

    plt.show()

def rot_matrix(miller_indices, rot_axis):
    print(f'Rotating {miller_indices} to align with axis {rot_axis}')
    u = miller_indices
    u /= np.linalg.norm(u)
    rot_axis /= np.linalg.norm(rot_axis)
    # Compute angle of rotation
    dot_prod = np.dot(u, rot_axis)
    #print(dot_prod)
    theta = np.arccos(dot_prod)
    #print(f'theta={theta}')
    
    # Compute axis of rotation
    r = np.cross(u, rot_axis)
    r /= np.linalg.norm(r) 

    r_shear = np.array([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]])
    r_shear2 = r_shear @ r_shear

    R = np.eye(3) + np.sin(theta)*r_shear + (1- np.cos(theta))*r_shear2
    #print(f'det(R) = {np.linalg.det(R)}')
    #print(f'rotation matrix to align {miller_indices} with axis {rot_axis} is \n {R}')
    #print(f'inverse matrix {np.linalg.inv(R)}')

    return R

# Rotate the lattice so that a particular crystallographic direction is aligned with the 'z'
# axis in the original coordinate system.
def rotate_lattice(points, direction, align_axis):
    miller_indices = np.array(direction, dtype=np.float64)
    rot_axis = np.array(align_axis, dtype=np.float64)
    R = rot_matrix(miller_indices, rot_axis)

    for i in range(points.shape[0]):
        vec = points[i,:]
        new_vec = R @ vec
        points[i,:] = new_vec

    return points

def generate_lattice(lattice_vectors, rep_x, rep_y, rep_z):
    a1, a2, a3 = lattice_vectors
    points = []
    for i in range(-rep_x, rep_x+1):
        for j in range(-rep_y, rep_y + 1):
            for k in range(-rep_z, rep_z+1):
                points.append(i*a1 + j*a2 + k*a3)
    return np.array(points)

def select_atoms_by_plane(points, z, tol=1e-5):
    points_in_plane = []
    for point in points:
        if (np.abs(point[2] - z) < tol):
            points_in_plane.append(point)

    return np.array(points_in_plane)

def select_atom_by_coordinate(lattice_points, x, y, z, tol=1e-5):
    for point in lattice_points:
        if (np.abs(point[0] - x) < tol and np.abs(point[1] - y) < tol and np.abs(point[2] - z) < tol):
            return point

def return_2d_vectors(points):
    return np.array([point[0], point[1]] for point in points)

def sort_by_distance(points, x, y):
    return np.array(sorted(points, key=lambda p: np.sqrt((p[0] - x)**2 + (p[1] - y)**2)))

num_reps = 3
rep_x = rep_y = rep_z = num_reps
a = 1.0

a1 = np.array([0, a/2, a/2])
a2 = np.array([a/2, 0, a/2])
a3 = np.array([a/2, a/2, 0])
lattice_vectors = [a1, a2, a3]

# Generate a lattice and rotate it such that the z axis in the 
# figure is aligned with a particular crystallographic direction
lattice_points = generate_lattice(lattice_vectors, rep_x, rep_y, rep_z)
rotated_points = rotate_lattice(lattice_points, [1, 1, 1], [0,0,1])

z0_plane = select_atoms_by_plane(rotated_points, z=0)
in_plane_vectors = return_2d_vectors(z0_plane)
sorted_vectors = sort_by_distance(z0_plane, 0.0, 0.0)

min_length = np.linalg.norm(sorted_vectors[1,:])
shortest_vectors = []

for point in sorted_vectors:
    if np.isclose(np.linalg.norm(point), min_length):
        shortest_vectors.append(point)
shortest_vectors = np.array(shortest_vectors)

for point in shortest_vectors:
    print(f"(x, y, z) ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}): r = {np.sqrt(point[0]**2 + point[1]**2):.4f}")


# go back to the original coordinate system.
print("Returning to original coordinate system")
new_old_vectors = sort_by_distance(rotate_lattice(shortest_vectors, [0,0,1], [1, 1, 1]), 0.0, 0.0)
for point in new_old_vectors:
    print(f"(x, y, z) ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}): r = {np.sqrt(point[0]**2 + point[1]**2):.4f}")
