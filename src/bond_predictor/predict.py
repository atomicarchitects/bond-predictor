import argparse

import numpy as np
import torch
from rdkit import Chem
import e3tools

from bond_predictor import rdkit_utils


def main():
    parser = argparse.ArgumentParser(description='Predict bond type')
    parser.add_argument('input', type=str, help='Input .xyz file')
    parser.add_argument('output', type=str, help='Output file')
    args = parser.parse_args()

    # Load molecule.
    mol = Chem.MolFromXYZFile(args.input)
    if mol is None:
        raise ValueError('Failed to load molecule')

    # Load network.
    out_path = 'bond_predictor.pt2'
    model = torch._inductor.aoti_load_package(out_path)

    # Predict bonds.
    mol_with_bonds = predict(model, mol)
    Chem.MolToPDBFile(mol_with_bonds, args.output)


def predict(model, mol):

    # Get atom features.
    coordinates = rdkit_utils.get_coordinates(mol)
    atom_types = rdkit_utils.get_atom_types(mol)

    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    atom_types = torch.tensor(atom_types, dtype=torch.int64)

    edge_index = e3tools.radius_graph(coordinates, r=10.0)
    bond_logits, charge_logits = model((coordinates, atom_types, edge_index))

    bond_types = torch.argmax(bond_logits, dim=-1)
    charges = torch.argmax(charge_logits, dim=-1) - 6

    bond_types = bond_types.cpu().numpy()
    charges = charges.cpu().numpy()
    edge_index = edge_index.cpu().numpy()

    # Add bonds to molecule, across all edges.
    mol = Chem.RWMol(mol)
    for (i, j), bond_type in zip(edge_index.T, bond_types):
        if i >= j:
            continue

        if bond_type == 0:
            continue
        if bond_type == 1:
            bond_order = Chem.rdchem.BondType.SINGLE
        elif bond_type == 2:
            bond_order = Chem.rdchem.BondType.DOUBLE
        elif bond_type == 3:
            bond_order = Chem.rdchem.BondType.TRIPLE
        elif bond_type == 4:
            bond_order = Chem.rdchem.BondType.AROMATIC
        else:
            raise ValueError('Invalid bond type')
        
        mol.AddBond(int(i), int(j), bond_order)

    return mol