import sys
import argparse
import numpy as np
import ase.io
from ase.atoms import Atoms
from ase.build import make_supercell

"""This is a version to search from twist angle applied to vdWH by optical analogue approximation."""

"""please check New Journal of Physics 16 (2014) 083028 for theoretical background"""

class Pyatoms():
    def __init__(self,vectors, positions, atomic_numbers):
        self.vectors = vectors
        self.positions = positions
        self.atomic_numbers = atomic_numbers

class theta():
    def __init__(self, radian, degree):
        self.radian = radian
        self.degree = degree

def calc_vectors_dis(v1):
    """calculate the angle between two vectors"""
    length_v1 = np.sqrt(np.dot(v1, v1))
    return abs(length_v1)

FLOAT_PREC = 1e-6
def float_eq(f1, f2, prec=FLOAT_PREC):
    """float equal"""
    return abs(f1 - f2) < prec

def exit(contect='[error] Unknown: Something goes wrong...'):
    """exit the program with some error msg."""
    print(contect)
    sys.exit(1)    
    
def ase_atoms_to_py_atoms(atoms: ase.atoms.Atoms) -> Pyatoms:
    vectors = np.array(atoms.cell.copy())
    positions = atoms.positions
    atomic_numbers = atoms.numbers
    atoms = Pyatoms(vectors, positions, atomic_numbers)
    return atoms

def py_atoms_to_ase_atoms(atoms: Pyatoms) -> ase.atoms.Atoms:
    vectors = [[j for j in k] for k in atoms.vectors]
    positions = [[j for j in k] for k in atoms.positions]
    numbers = [i for i in atoms.atomic_numbers]
    atoms = Atoms(
        numbers=numbers, positions=positions, cell=vectors, pbc=[True, True, True]
    )
    return atoms    
    
def check_vectors(atoms: ase.atoms.Atoms):
    len_ang = atoms.cell.cellpar()
    if not (float_eq(len_ang[3], 90.0) and
            float_eq(len_ang[4], 90.0)):
        exit('[error] Input structures: c axis must be in z direction')
    if not (float_eq(len_ang[0], len_ang[1]) and
            float_eq(len_ang[5], 120)):
        exit('[error] This version is only for P6 (a=b & γ=120)')

def coord_cart2frac(cell_vecs, cart_vec):
    """Transfrom the cart coords to frac coords"""
    cell_vecs_inv = np.linalg.inv(cell_vecs)
    frac_vec = np.dot(cart_vec, cell_vecs_inv)
    return frac_vec

def coord_frac2cart(cell_vecs, frac_vec):
    """Transfrom the frac coords to cart coords"""
    return np.dot(frac_vec, cell_vecs)

def get_supercell_vecs(trans_2D, unit_vecs, super_z):
    """Get the supercell lattice"""
    super_2d = np.dot(trans_2D, unit_vecs[0:2,0:2])
    supercell_vecs = np.array([[super_2d[0,0], super_2d[0,1], 0],
                               [super_2d[1,0], super_2d[1,1], 0],
                               [0, 0, super_z]])
    return supercell_vecs

def lattice_points_in_supercell(supercell_matrix): #copy from ASE
    """Find all lattice points contained in a supercell.

    Adapted from pymatgen, which is available under MIT license:
    The MIT License (MIT) Copyright (c) 2011-2012 MIT & The Regents of the
    University of California, through Lawrence Berkeley National Laboratory
    """

    diagonals = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[
        np.all(frac_points < 1 - 1e-10, axis=1)
        & np.all(frac_points >= -1e-10, axis=1)
    ]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects

def get_supercell_atoms_pos(supercell_vecs: "np.array",
                            trans_2D: "np.array",
                            unit: ase.atoms.Atoms) -> ase.atoms.Atoms:
    trans_matrix = np.array([[trans_2D[0,0], trans_2D[0,1], 0],
                               [trans_2D[1,0], trans_2D[1,1], 0],
                               [0, 0, 1]])
    lattice_points_frac = lattice_points_in_supercell(trans_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell_vecs)
    superatoms = Atoms(cell=supercell_vecs, pbc=unit.pbc)
    for lp in lattice_points:
        shifted_atoms = unit.copy()
        shifted_atoms.positions += lp
        superatoms.extend(shifted_atoms)
    return superatoms

def coincidence(vectors_super, vectors_indiv,
                trans_2D: "np.array",
                unit: ase.atoms.Atoms)-> Pyatoms:
    superatoms = get_supercell_atoms_pos(vectors_indiv, trans_2D, unit)
    superatoms_frac = coord_cart2frac(vectors_indiv, superatoms.positions)
    superatoms_pos = coord_frac2cart(vectors_super, superatoms_frac)
    return Pyatoms(vectors_super, superatoms_pos, superatoms.numbers)

def gen_supercell(bottom: ase.atoms.Atoms,
                  top: ase.atoms.Atoms,
                  theta: float, 
                  super_z: float,
                  delta_z: float) -> ase.atoms.Atoms :
    """get transition matrix"""
    lc_ratio = bottom.cell.cellpar()[0]/top.cell.cellpar()[0]
    theta_rad = theta/180*np.pi
    deno = np.sqrt(1 + np.square(lc_ratio) - 2 * lc_ratio * np.cos(theta_rad))
    phi_moire = np.arccos((lc_ratio * np.cos(theta_rad) -1)/deno)
    m = 1/np.sqrt(3) /deno * np.sin(phi_moire) + 1/deno * np.cos(phi_moire)
    n = 2/np.sqrt(3) /deno * np.sin(phi_moire)
    r = 1/np.sqrt(3) /deno * lc_ratio * np.sin(phi_moire-theta_rad) + 1/deno * lc_ratio * np.cos(phi_moire-theta_rad)
    s = 2/np.sqrt(3) /deno * lc_ratio * np.sin(phi_moire-theta_rad)
    para_matrx = np.array([m,n,r,s])
    para_matrx = np.around(para_matrx)
    tran_2D_b = np.array([[-para_matrx[0], -para_matrx[1]],[para_matrx[1], para_matrx[1]-para_matrx[0]]]) #to keep lattice_x positive
    tran_2D_t = np.array([[-para_matrx[2], -para_matrx[3]],[para_matrx[3], para_matrx[3]-para_matrx[2]]])
    """get supercell lattice"""
    vectors_b = get_supercell_vecs(tran_2D_b, bottom.cell, super_z)
    top_rot = top.copy()
    top_rot.rotate(theta, 'z', rotate_cell=True)
    vectors_t = get_supercell_vecs(tran_2D_t, top_rot.cell, super_z)
    vectors = (vectors_b + vectors_t)/2
    Delta = calc_vectors_dis(vectors_t[0,:2] - vectors_b[0,:2])/calc_vectors_dis(vectors[0,:2])
    #mismatch = (calc_vectors_dis(vectors_t[0,:2]) - calc_vectors_dis(vectors_b[0,:2]))/calc_vectors_dis(vectors[0,:2])
    """get atom position"""
    bottom_super = coincidence(vectors, vectors_b, tran_2D_b, bottom)
    top_super = coincidence(vectors, vectors_t, tran_2D_t, top_rot)
    top_super.positions[..., 2] = top_super.positions[..., 2] + delta_z
    positions = np.vstack((bottom_super.positions, top_super.positions))
    number = np.hstack((bottom_super.atomic_numbers, top_super.atomic_numbers))
    results = py_atoms_to_ase_atoms(Pyatoms(vectors, positions, number))
    return results, Delta

def get_parser():
    parser = argparse.ArgumentParser(description="A simply script for heterostures bilayers from given transformation matrix, can not search from twist angle")
    parser.add_argument('-b','--bottom', required=True, help='Path to lower layer, need to be recognized by ASE')
    parser.add_argument('-t','--top', required=True, help='Path to upper layer, need to be recognized by ASE')
    parser.add_argument('-a','--angle', help='Multi twist angle',nargs='+', type=float)
    parser.add_argument('-al','--alist', help='start end step',nargs='+', type=float)
    parser.add_argument('-tol','--tolerance', help='Delta/moire_lattice (%), default = 1%', type=float, default = 1.0)
    parser.add_argument('-z', help='super lattice of z direction, default = 60 A', type=float, default = 60.0)
    parser.add_argument('-d', help='distance of two layers, default = 7 A', type=float, default = 7.0)
    parser.add_argument('-w','--write', help='Path to write supercell, need to be recognized by ASE')
    parser.add_argument('-o','--outformat', help='output file-format, like lammps-data', default = None)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    bottom = ase.io.read(args.bottom)
    top = ase.io.read(args.top)
    check_vectors(bottom)
    check_vectors(top)
    if args.alist == None:
        for ang in args.angle:
            result, Delta = gen_supercell(bottom, top, ang, args.z, args.d)
            num = len(result)
            formula=result.get_chemical_formula()
            if Delta*100 < args.tolerance:
                print("Delta: {1:.2f}, angel: {0:.2f}, num: {2:}".format(ang, Delta*100, num))
                if args.write == None:
                    result.write("Super_{0:.2f}_{1}_{2}.xsf".format(ang,num,formula))
                else:
                    result.write(args.write, format=args.outformat)
    else:
        for ang in np.arange(args.alist[0], args.alist[1], args.alist[2]):
            result, Delta = gen_supercell(bottom, top, ang, args.z, args.d)
            num = len(result)
            formula=result.get_chemical_formula()
            if Delta*100 < args.tolerance:
                print("Delta: {1:.2f}, angel: {0:.2f}, num: {2:}".format(ang, Delta*100, num))
                if args.write == None and Delta < 1.0:
                    result.write("Super_{0:.2f}_{1}_{2:.2f}.xsf".format(ang,num,Delta*100))
                else:
                    result.write(args.write, format=args.outformat)
