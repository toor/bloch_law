import numpy as np

lattice_types = [
    "sc_100",
    "bcc_100",
    "bcc_110",
    "bcc_111",
    "fcc_100",
    "fcc_110",
    "fcc_111",
    # Simple tetragonal
    "st_100",
    "st_110",
    "st_001",
    # Body-centred tetragonal
    "bct_100",
    "bct_010",
    "bct_001",
    # Simple orthorhombic
    "so_100",
    "so_010",
    "so_001",
    # Base-centred orthorhombic is the same as simple orthorhombic
    "baco_100",
    "baco_010",
    "baco_001",
    # Body-centred orthorhombic
    "bco_100",
    "bco_010",
    "bco_001",
    "bco_110",
    "bco_111",
    # Hexagonal
    "sh_100",
    "sh_010",
    "sh_110",
    "sh_001",
    # Rhombohedral (trigonal)
    "rh_111",
    "rh_110",
    # Monoclinic
    "mc_100",
    "mc_001"
]

rule all:
    input: expand("lattices/{lattice_type}/projection.png", lattice_type=lattice_types),
           expand("lattices/{lattice_type}/nn_vectors.dat", lattice_type=lattice_types)

rule generate:
    output: "lattices/{lattice_type}/projection.png",
            "lattices/{lattice_type}/nn_vectors.dat"
    params:
        a = 1.0
        b = 1.0
        c = 2.0
        alpha = np.pi/
