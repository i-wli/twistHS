import argparse
import numpy as np
import ase.io
from src.func import *

"""please check 2012 J. Phys.: Condens. Matter 24 314210 for theoretical background"""

def coincidence(vectors_super, vectors_indiv,
                trans_2D: "np.array",
                unit: ase.atoms.Atoms)-> Pyatoms:
    superatoms = get_supercell_atoms_pos(vectors_indiv, trans_2D, unit)
    superatoms_frac = coord_cart2frac(vectors_indiv, superatoms.positions)
    superatoms_pos = coord_frac2cart(vectors_super, superatoms_frac)
    return Pyatoms(vectors_super, superatoms_pos, superatoms.numbers)

def get_trans_matrix_2d(alpha_value, omega, q, p1, p2):
    """get transformation matrix with constant:
    alpha_value: twist angle
    omega: lattice angle (radian)
    q = Ro1/Ro2
    p1 = R'o1/Ro1"""

    Delta = 1 + p1*p2 - (p1 + p2)*np.cos(alpha_value) + (p1 - p2)*np.cos(omega)*np.sin(alpha_value)

    # Define the matrix that maps Ro to Ro'
    transformation_matrix_R_prime = np.array([p1*np.array([np.sin(omega - alpha_value), q*np.sin(alpha_value)]),
                                            p2*np.array([-1/q*np.sin(alpha_value), np.sin(omega + alpha_value)])])
    transformation_matrix_R_prime = 1 / np.sin(omega) * transformation_matrix_R_prime

    # Define the matrix that maps Ro to RM
    transformation_matrix_RM = np.array([p1*np.array([(np.sin(omega - alpha_value) - p2*np.sin(omega)), q*np.sin(alpha_value)]),
                                        p2*np.array([-1/q*np.sin(alpha_value), (np.sin(omega + alpha_value) - p1*np.sin(omega))])])
    transformation_matrix_RM = 1 / (Delta * np.sin(omega)) * transformation_matrix_RM

    # The mapping matrix from Ro' to RM
    mapping_matrix_R_prime_to_RM = np.linalg.inv(transformation_matrix_R_prime).dot(transformation_matrix_RM)
    return transformation_matrix_RM, mapping_matrix_R_prime_to_RM


def gen_supercell(bottom: ase.atoms.Atoms,
                  top: ase.atoms.Atoms,
                  theta: float, 
                  super_z: float,
                  delta_z: float) -> ase.atoms.Atoms :
    """get parameters"""
    q = bottom.cell.cellpar()[0]/bottom.cell.cellpar()[1]
    p1 = top.cell.cellpar()[0]/bottom.cell.cellpar()[0]
    p2 = top.cell.cellpar()[1]/bottom.cell.cellpar()[1]
    omega = bottom.cell.cellpar()[5]
    if float_eq(omega, top.cell.cellpar()[5], prec=1e-3) == False:
        exit('[error] the angle of two layers are different')
    if p1 == 1 and p2 == 1:
        exit('[error] homobilayers are usually gnereated by another way: you can check the my branch "old" for the code')

    theta_rad = np.radians(theta)
    omega = np.radians(omega)
    tran_2D_b, tran_2D_t = get_trans_matrix_2d(theta_rad, omega, q, p1, p2)
    tran_2D_b = -np.round(tran_2D_b)
    tran_2D_t = -np.round(tran_2D_t)
    #print(tran_2D_b, tran_2D_t)

    """get supercell lattice"""
    vectors_b = get_supercell_vecs(tran_2D_b, bottom.cell, super_z)
    top_rot = top.copy()
    top_rot.rotate(theta, 'z', rotate_cell=True)
    vectors_t = get_supercell_vecs(tran_2D_t, top_rot.cell, super_z)
    vectors = (vectors_b + vectors_t)/2
    Delta = calc_vectors_dis(vectors_t[0,:2] - vectors_b[0,:2])/calc_vectors_dis(vectors[0,:2])

    """get atom position"""
    bottom_super = coincidence(vectors, vectors_b, tran_2D_b, bottom)
    top_super = coincidence(vectors, vectors_t, tran_2D_t, top_rot)
    top_super.positions[..., 2] = top_super.positions[..., 2] + delta_z
    positions = np.vstack((bottom_super.positions, top_super.positions))
    number = np.hstack((bottom_super.atomic_numbers, top_super.atomic_numbers))
    results = py_atoms_to_ase_atoms(Pyatoms(vectors, positions, number))
    return results, Delta

def get_parser():
    parser = argparse.ArgumentParser(description="Generate twisted van der Waals heterostructures.")
    parser.add_argument('-b','--bottom', required=True, help='Path to lower layer, need to be recognized by ASE')
    parser.add_argument('-t','--top', required=True, help='Path to upper layer, need to be recognized by ASE')
    parser.add_argument('-a','--angle', help='Multi twist angle',nargs='+', type=float)
    parser.add_argument('-al','--alist', help='start end step',nargs='+', type=float)
    parser.add_argument('-tol','--tolerance', help='Delta/moire_lattice (%%), default = 1%%', type=float, default = 1.0)
    parser.add_argument('-z', help='superlattice constant along z direction, default = 30 A', type=float, default = 30.0)
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
                if args.write == None:
                    result.write("Super_{0:.2f}_{1}_{2}.xsf".format(ang,num,formula))
                else:
                    result.write(args.write, format=args.outformat)
