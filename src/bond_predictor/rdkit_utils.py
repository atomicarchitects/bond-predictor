import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def get_atom_features(atom: Chem.Atom) -> int:
    """
    Get atom features as a simple one-hot encoding of atom type

    Args:
        atom: RDKit atom object

    Returns:
        One-hot encoded atom type (int)
    """
    return atom.GetAtomicNum()


def get_coordinates(mol: Chem.Mol):
    """
    Get atom coordinates from an RDKit molecule

    Args:
        mol: RDKit molecule

    Returns:
        coordinates: Numpy array of atom 3D coordinates (num_atoms, 3)
    """
    # Get conformer
    conf = mol.GetConformer()

    # Extract coordinates
    num_atoms = mol.GetNumAtoms()
    coordinates = np.zeros((num_atoms, 3))

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coordinates[i] = [pos.x, pos.y, pos.z]

    return coordinates


def get_atom_types(mol: Chem.Mol):
    """
    Get atom types from an RDKit molecule

    Args:
        mol: RDKit molecule

    Returns:
        atom_types: Numpy array of atom types (num_atoms,)
    """
    num_atoms = mol.GetNumAtoms()
    atom_types = np.zeros(num_atoms, dtype=int)

    for i in range(num_atoms):
        atom_types[i] = get_atom_features(mol.GetAtomWithIdx(i))

    return atom_types


def embed_molecule(mol: Chem.Mol, add_H: bool = True):
    """
    Embed an RDKit molecule (if necessary) and extract atom coordinates and bond types

    Args:
        mol: RDKit molecule
        add_H: Whether to add hydrogen atoms to the molecule

    Returns:
        coordinates: Numpy array of atom 3D coordinates (num_atoms, 3)
        atom_types: Numpy array of atom types (num_atoms,)
        bond_matrix: Numpy array of bond types (num_atoms, num_atoms)
            0: No bond
            1: Single bond
            2: Double bond
            3: Triple bond
            4: Aromatic bond
    """
    # Add hydrogens if requested
    if add_H:
        mol = Chem.AddHs(mol)

    # Generate 3D coordinates if not present
    if not mol.GetNumConformers():
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.useBasicKnowledge = True
        params.enforceChirality = True

        # Generate and optimize conformer
        AllChem.EmbedMolecule(mol, params)

    # Get conformer
    conf = mol.GetConformer()

    # Extract coordinates
    num_atoms = mol.GetNumAtoms()
    coordinates = np.zeros((num_atoms, 3))
    atom_types = np.zeros(num_atoms, dtype=int)

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coordinates[i] = [pos.x, pos.y, pos.z]
        atom_types[i] = get_atom_features(mol.GetAtomWithIdx(i))

    # Create bond matrix
    bond_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()

        # Map bond types to integers
        if bond_type == 1.0:
            bond_idx = 1  # Single bond
        elif bond_type == 2.0:
            bond_idx = 2  # Double bond
        elif bond_type == 3.0:
            bond_idx = 3  # Triple bond
        elif bond_type == 1.5:
            bond_idx = 4  # Aromatic bond
        else:
            bond_idx = 1  # Default to single bond

        # Set bond type (symmetric matrix)
        bond_matrix[i, j] = bond_idx
        bond_matrix[j, i] = bond_idx

    # Get charges
    charges = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    return coordinates, atom_types, bond_matrix, charges


def embed_molecule_for_torch(mol: Chem.Mol, add_H: bool = True):
    """
    Embed an RDKit molecule (if necessary) and extract atom coordinates and bond types, returning torch tensors

    Args:
        mol: RDKit molecule
        add_H: Whether to add hydrogen atoms to the molecule

    Returns:
        coordinates: Torch tensor of atom 3D coordinates (batch_size, num_atoms, 3)
        atom_types: Torch tensor of atom types (batch_size, num_atoms)
        bond_matrix: Torch tensor of bond types (batch_size, num_atoms, num_atoms)
    """
    coordinates, atom_types, bond_matrix, charges = embed_molecule(mol, add_H=add_H)

    # Convert to torch tensors with batch dimension
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    atom_types = torch.tensor(atom_types, dtype=torch.long)
    bond_matrix = torch.tensor(bond_matrix, dtype=torch.long)
    charges = torch.tensor(charges, dtype=torch.float32)

    return coordinates, atom_types, bond_matrix, charges
