import os.path as osp
import numpy as np
import torch
from indigo import Indigo, IndigoObject
from mendeleev import element
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm


def calc_atoms_info(accepted_atoms: tuple) -> dict:
    """
    Given a tuple of accepted atoms, return a dictionary with the atom symbol as the key and a tuple of
    the period, group, subshell, and number of electrons as the value.

    :param accepted_atoms: tuple of strings
    :type accepted_atoms: tuple
    :return: A dictionary with the atomic number as the key and the period, group, shell, and number of
    electrons as the value.
    """
    mendel_info = {}
    shell_to_num = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
    for atom in accepted_atoms:
        mendel_atom = element(atom)
        period = mendel_atom.period
        group = mendel_atom.group_id
        shell, electrons = mendel_atom.ec.last_subshell()
        mendel_info[atom] = (period, group, shell_to_num[shell[1]], electrons)
    return mendel_info


# This function should be optimized so that we do not iterate through atoms
def is_atom_in_ring(molecule: IndigoObject, atom_index: int):
    """
    Determine if a specific atom in a given molecule is part of a ring.

    Parameters:
    - molecule (IndigoObject): Molecule to check.
    - atom_index (int): Index of the atom to check.

    Returns:
    - int: Returns 0 if the atom is in a ring, 1 otherwise.
    """
    ring_atoms = set()

    for ring in molecule.iterateRings(0, molecule.countAtoms()):
        for atom in ring.iterateAtoms():
            ring_atoms.add(atom.index())

    return 1 if atom_index in ring_atoms else 0


def atom_to_vector(atom: IndigoObject, mendel_info: dict, molecule: IndigoObject):
    """
    Given an atom, return a vector of length 8 with the following information:

    1. Atomic number
    2. Period
    3. Group
    4. Number of electrons + atom's charge
    5. Shell
    6. Total number of hydrogens
    7. Whether the atom is in a ring
    8. Number of neighbors

    :param atom: atom IndigoObject
    :param mendel_info: a dictionary of the form {'C': (3, 1, 1, 2), 'O': (3, 1, 1, 2), ...}
    :param molecule: molecule IndigoObject
    :type mendel_info: dict
    :return: The vector of the atom.
    """

    vector = np.zeros(8, dtype=np.int8)
    period, group, shell, electrons = mendel_info[atom.symbol()]
    vector[0] = atom.atomicNumber()
    vector[1] = period
    vector[2] = group
    vector[3] = electrons + atom.charge()
    vector[4] = shell
    vector[5] = atom.countHydrogens()
    vector[6] = is_atom_in_ring(molecule, atom.index())
    neighbors = [neighbor for neighbor in atom.iterateNeighbors()]
    vector[7] = len(neighbors)
    return vector


def graph_to_atoms_vectors(molecule: IndigoObject, max_atoms: int, mendel_info: dict):
    """
        Given a molecule, it returns a vector of shape (max_atoms, 11) where each row is an atom and each
        column is a feature.

        :param molecule: The molecule to be converted to a vector
        :type molecule: IndigoObject
        :param max_atoms: The maximum number of atoms in the molecule
        :type max_atoms: int
        :param mendel_info: a dictionary containing the information about the Mendel system
        :type mendel_info: dict
        :return: The atoms_vectors array
    """
    atoms_vectors = np.zeros((max_atoms, 8), dtype=np.int8)
    for atom in molecule.iterateAtoms():
        atoms_vectors[atom.index()] = atom_to_vector(atom, mendel_info, molecule)

    return atoms_vectors


def graph_to_coo_matrix(molecule):
    mol_adj, edge_attr = [], []
    for bond in molecule.iterateBonds():
        source = bond.source().index()
        dest = bond.destination().index()
        mol_adj.append([source, dest])
        edge_attr.append(bond.bondOrder())
    return torch.tensor(mol_adj, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.long)


def preprocess_molecules(file):
    accepted_atoms = ('C', 'N', 'S', 'O', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'P', 'Si')
    mendel_info = calc_atoms_info(accepted_atoms)
    indigo = Indigo()
    with open(file) as bbs:
        for line in bbs:
            smiles = line.strip()
            molecule = indigo.loadMolecule(smiles)
            molecule.dearomatize()
            mol_adj, edge_attr = graph_to_coo_matrix(molecule)
            mol_atoms_x = graph_to_atoms_vectors(molecule, molecule.countAtoms(), mendel_info)
            mol_pyg_graph = Data(
                x=torch.tensor(mol_atoms_x, dtype=torch.int8),
                edge_index=mol_adj.t().contiguous(),
                edge_attr=edge_attr
            )
            mol_pyg_graph = ToUndirected()(mol_pyg_graph)
            assert mol_pyg_graph.is_undirected()
            yield mol_pyg_graph


def process_molecules(file_name, name, processed_dir):
    processed_molecules = []
    for data in tqdm(preprocess_molecules(file_name)):
        processed_molecules.append(data)
    torch.save(processed_molecules, osp.join(processed_dir, f"{name}_molecules.pt"))


