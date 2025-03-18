import numpy as np
import matplotlib.pyplot as plt

import sys
import os

def gen_rhombohedral(a, alpha):
    return a*np.array([[1, 0, 0],
                       [np.cos(alpha), np.sin(alpha), 0],
                       [np.cos(alpha), (1-np.cos(alpha))*np.cos(alpha)/np.sin(alpha), np.sqrt((1-np.cos(alpha))(1 + 2*np.cos(alpha))/(1 + np.cos(alpha)))]], dtype=np.float64) 
def gen_triclinic(a, b, c, alpha, beta, gamma): 
    return np.array([[a, 0, 0],
                    [b*np.cos(gamma), b*np.sin(gamma), 0],
                    [c*np.cos(beta),
                     c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma),
                     c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma))**2)]], dtype=np.float64)

def get_basis_vectors(lattice, a, b, c, alpha, beta, gamma):

    match lattice:
        case "cP":
            return (np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=np.float64))
        case "cI":
            return ((a/2)*np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=np.float64))
        case "cF":
            return ((a/2)*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=np.float64))
        case "tP":
            return (np.array([[a, 0, 0], [0, a, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0]], dtype=np.float64))
        case "tI":
            return (np.array([[a, -a, c], [a, a, c], [-a, -a, c]], dtype=np.float64) / 2, np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=np.float64))
        case "oP":
            return (np.array([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]], dtype=np.float64))
        case "oS":
            return (np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]], dtype=np.float64)) 
        case "oI":
            return (np.array([[a, b, c], [-a, b, c], [a, -b, c]], dtype=np.float64) / 2, np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=np.float64))
        case "oF":
            return (np.array([[0, b, c], [a, 0, c], [a, b, 0]], dtype=np.float64) / 2, np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]], dtype=np.float64))
        case "hP":
            return (np.array([[a, 0, 0], [-a/2, a*np.sqrt(3)/2, 0], [0, 0, c]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0]], dtype=np.float64))
        case "hR":
            return (gen_rhombohedral(a, alpha), np.array([[1,1,1], [1,1,0]], dtype=np.float64))
        case "mP":
            return (np.array([[a, 0, 0], [0, b, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64))
        case "mC":
            return (np.array([[a/2, b/2, 0], [-a/2, b/2, 0], [c*np.cos(beta), 0, c*np.sin(beta)]], dtype=np.float64), np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64))
        case "aP":
            return (gen_triclinic(a, b, c, alpha, beta, gamma), None)

    #def get_bravais_lattices(a, b, c, alpha, beta, gamma, theta): 
    #    lattices = {
    #        "sc": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([0, a, 0]),
    #            "a3": np.array([0, 0, a])
    #        },
    #        "bcc": {
    #            "a1": (a/2) * np.array([-1,  1,  1]),
    #            "a2": (a/2) * np.array([ 1, -1,  1]),
    #            "a3": (a/2) * np.array([ 1,  1, -1])
    #        },
    #        "fcc": {
    #            "a1": (a/2) * np.array([0,  1,  1]),
    #            "a2": (a/2) * np.array([1,  0,  1]),
    #            "a3": (a/2) * np.array([1,  1,  0])
    #        },
    #        "st": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([0, a, 0]),
    #            "a3": np.array([0, 0, c])
    #        },
    #        "bct": {
    #            "a1": 0.5 * np.array([ a, -a,  c]),
    #            "a2": 0.5 * np.array([ a,  a,  c]),
    #            "a3": 0.5 * np.array([-a, -a,  c])
    #        },
    #        "sco": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([0, b, 0]),
    #            "a3": np.array([0, 0, c])
    #        },
    #        "baco": {
    #            "a1": 0.5 * np.array([ a,  b, 0]),
    #            "a2": 0.5 * np.array([-a,  b, 0]),
    #            "a3": np.array([0, 0, c])
    #        },
    #        "bco": {
    #            "a1": 0.5 * np.array([ a,  b,  c]),
    #            "a2": 0.5 * np.array([-a,  b,  c]),
    #            "a3": 0.5 * np.array([ a, -b,  c])
    #        },
    #        "fco": {
    #            "a1": 0.5 * np.array([0,  b,  c]),
    #            "a2": 0.5 * np.array([a,  0,  c]),
    #            "a3": 0.5 * np.array([a,  b, 0])
    #        },
    #        "sh": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([-a/2, (a*np.sqrt(3))/2, 0]),
    #            "a3": np.array([0, 0, c])
    #        },
    #        "rh": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([a*np.cos(theta), a*np.sin(theta), 0]),
    #            "a3": np.array([
    #                a*np.cos(theta),
    #                a*(np.cos(theta) - np.cos(theta)**2 + np.sin(theta)**2) / np.sin(theta),
    #                a*np.sqrt(1 - 3*np.cos(theta)**2 + 2*np.cos(theta)**3) / np.sin(theta)
    #            ])
    #        },
    #        "mc": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([0, b, 0]),
    #            "a3": np.array([c*np.cos(beta), 0, c*np.sin(beta)])
    #        },
    #        "bmc": {
    #            "a1": 0.5 * np.array([a, b, 0]),
    #            "a2": 0.5 * np.array([-a, b, 0]),
    #            "a3": np.array([c*np.cos(beta), 0, c*np.sin(beta)])
    #        },
    #        "tric": {
    #            "a1": np.array([a, 0, 0]),
    #            "a2": np.array([b*np.cos(gamma), b*np.sin(gamma), 0]),
    #            "a3": np.array([
    #                c*np.cos(beta),
    #                c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma),
    #                c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma))**2)
    #            ])
    #        }
    #    }
    #    return lattices

def plot_plane(fig, ax, plane_z):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y)
    Z = np.full_like(X, plane_z)
    
    ax.plot_surface(X, Y, Z, color='red', alpha=0.1)

def plot_lattice(points, filename, vectors=None):
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
            #print(f"(x, y, z) ({vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}): r = {np.sqrt(vec[0]**2 + vec[1]**2):.4f}")
            ax.quiver(0.0, 0.0, 0.0, vec[0], vec[1], vec[2], color='red', arrow_length_ratio=0.2)

    ax.set_xticks(ticks=[0.0, 1.0, -1.0], labels=['x=0', '1', '-1'])
    ax.set_yticks(ticks=[0.0, 1.0, -1.0], labels=['y=0', '1', '-1'])
    ax.set_zticks(ticks=[0.0, 1.0, -1.0], labels=['z=0', '1', '-1'])

    fig.savefig(filename)
    plt.close()

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
    miller_indices = direction
    rot_axis = align_axis
    R = rot_matrix(miller_indices, rot_axis)

    for i in range(points.shape[0]):
        vec = points[i,:]
        new_vec = R @ vec
        points[i,:] = new_vec

    return points

def generate_lattice(lattice_vectors, rep_x, rep_y, rep_z):
    print(lattice_vectors)
    a1 = lattice_vectors[0,:]
    a2 = lattice_vectors[1,:]
    a3 = lattice_vectors[2,:]
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

def in_plane_vectors_original_basis(lattice, plane, miller_indices, rot_axis):
    z0_plane = select_atoms_by_plane(lattice, plane)
    sorted_vectors = sort_by_distance(z0_plane, 0.0, 0.0)

    min_length = np.linalg.norm(sorted_vectors[1,:])
    shortest_vectors = []
    
    for point in sorted_vectors:
        if np.isclose(np.linalg.norm(point), min_length):
            shortest_vectors.append(point)

    shortest_vectors = rotate_lattice(np.array(shortest_vectors), rot_axis, miller_indices)

    return return_2d_vectors(shortest_vectors)

def calculate_in_plane_vectors(lat, a, b, c, alpha, beta, gamma, prefix):
    rep_x = rep_y = rep_z = 3
    z_axis = np.array([0,0,1], dtype=np.float64)

    (basis_vectors, symmetry_directions) = get_basis_vectors(lat, a, b, c, alpha, beta, gamma)
    lattice = generate_lattice(basis_vectors, rep_x, rep_y, rep_z)
    image_file = prefix + "unrotated_lattice.pdf"
    plot_lattice(lattice, image_file, vectors=None)

    for i, _ in enumerate(symmetry_directions):
        print(f"Generating in-plane nearest neighbour vectors for lattice {lat} with plane defined by high-symmetry direction {symmetry_directions[i].astype(np.int64)}\n")
        axis = symmetry_directions[i]
        axis_string = str(int(axis[0] * 100 + axis[1]*10 + axis[0]*1))
        
        #print(f"Axis = {axis}, aligning to {z_axis}")
        # Rotate the lattice so that the given symmetry direction is aligned along 
        # the vertical axis of the figure
        if np.array_equal(axis, z_axis):
            #print("Axes are already aligned; skipping this symmetry.\n")
            z0_plane = select_atoms_by_plane(lattice, z=0)
        else:
            rotated_lattice = rotate_lattice(lattice, axis, z_axis)
            # Select all lattice points (atoms) that lie in the plane z = 0
            z0_plane = select_atoms_by_plane(rotated_lattice, z=0)
        
        # Sort these vectors by length and isolate the N shortest ones.
        sorted_vectors = sort_by_distance(z0_plane, 0.0, 0.0)
        print(sorted_vectors)
        min_length = np.linalg.norm(sorted_vectors[1,:])
        shortest_vectors = []
        for vec in sorted_vectors:
            if np.isclose(np.linalg.norm(vec), min_length):
                shortest_vectors.append(vec)
        shortest_vectors = np.array(shortest_vectors)
        # Again check if the symmetry axis is already along z.
        if np.array_equal(axis, z_axis):
            #print("Axes are already aligned; skipping this symmetry.\n")
            shortest_vectors_rot = shortest_vectors
        else:
            # Return to the original coordinate system
            shortest_vectors_rot = rotate_lattice(shortest_vectors, z_axis, axis)

        # Plot lattice projection and save image
        image_file = prefix + f"{axis_string}_lattice.pdf"
        dat_file = prefix + f"{axis_string}_nn_vectors.dat"
        plot_lattice(z0_plane, image_file, vectors=shortest_vectors)
        # Write nearest neighbour vectors to file.
        with open(dat_file, "w") as f:
            f.write("Index    x    y    z\n")
            for i, vec in enumerate(shortest_vectors_rot):
                f.write(f"{i}    {vec[0]}    {vec[1]}    {vec[2]}\n")
            f.close()

a = 1.0 
b = 1.5
c = 2.0
alpha = np.pi/4
beta = np.pi/3
gamma = np.pi/6

lattice_types = ["cP", "cI", "cF", "tP", "tI", "oP", "oS", "oI", "oF", "hP", "hR", "mP", "mS"]

for lat in lattice_types:
    prefix = f"nn_data_{lat}/"
    os.makedirs(os.getcwd() + "/" + prefix, exist_ok=True)

    calculate_in_plane_vectors(lat, a, b, c, alpha, beta, gamma, prefix)

sys.exit(0)



def main(lattice_type, alignment, a, b, c, alpha, beta, gamma, theta, filename):
    rep_x = rep_y = rep_z = 3
    miller_indices = alignment
    dic = get_bravais_lattices(a, b, c, alpha, beta, gamma, theta)[lattice_type]  
    lattice_vectors = [dic["a1"], dic["a2"], dic["a3"]]
    print(lattice_vectors)

    lattice = generate_lattice(lattice_vectors, rep_x, rep_y, rep_z)
    rotated_lattice = rotate_lattice(lattice, miller_indices, [0, 0, 1])
    z0_plane = select_atoms_by_plane(rotated_lattice, z=0)
    sorted_vectors = sort_by_distance(z0_plane, 0.0, 0.0)

    min_length = np.linalg.norm(sorted_vectors[1,:])
    shortest_vectors = []
    for vec in sorted_vectors:
        if np.isclose(np.linalg.norm(vec), min_length):
            shortest_vectors.append(vec)
            print(f"(x,y,z) = ({vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f})")
    shortest_vectors = np.array(shortest_vectors)
    print(f"Found {shortest_vectors.shape[0]} nearest neighbours for {lattice_type} lattice in [{alignment[0]}{alignment[1]}{alignment[2]}] plane.\n")

    plot_lattice(z0_plane, filename, vectors=shortest_vectors)
   


    sys.exit(0)

lattice_type = snakemake.params[0]
alignment = snakemake.params[1]
a = snakemake.params[2]
b = snakemake.params[3]
c = snakemake.params[4]
alpha = snakemake.params[5]
beta = snakemake.params[6]
gamma = snakemake.params[7]
theta = snakemake.params[8]
fig_file = snakemake.params[9]


main("fcc", [1,1,1], 1.0, 0, 0, 0, 0, 0, 0, "fcc_111_projection.png")


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
rotated_basis_vectors = rotate_lattice(np.array(lattice_vectors), [1, 1, 1], [0, 0, 1])

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

filename = "2.png"
#plot_lattice(lattice_points, a, vectors=new_old_vectors, plot_planes=False)
plot_lattice(z0_plane, filename, vectors=shortest_vectors)

# go back to the original coordinate system.
print("Returning to original coordinate system")
new_old_vectors = sort_by_distance(rotate_lattice(shortest_vectors, [0,0,1], [1, 1, 1]), 0.0, 0.0)
for point in new_old_vectors:
    print(f"(x, y, z) ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}): r = {np.sqrt(point[0]**2 + point[1]**2):.4f}")

