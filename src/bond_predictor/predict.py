import argparse

import os
from pathlib import Path

import torch
import tqdm
from rdkit import Chem
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import rdDepictor
import e3tools

from bond_predictor import rdkit_utils


def load_molecule(file_path: str) -> Chem.Mol:
    """
    Load a molecule from a file using RDKit based on the file extension.

    Parameters:
    file_path (str): Path to the molecular file

    Returns:
    RDKit molecule object or None if loading fails
    """
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Initialize molecule as None
    mol = None

    # Handle different file formats based on extension
    if file_extension in [".smi", ".smiles"]:
        # For SMILES files, read the first line
        with open(file_path, "r") as f:
            smiles = f.readline().strip().split()[0]  # Get the first token from the first line
            mol = Chem.MolFromSmiles(smiles)

    elif file_extension == ".mol":
        # For MDL MOL files
        mol = Chem.MolFromMolFile(file_path)

    elif file_extension == ".sdf":
        # For SDF files (returns the first molecule in the file)
        supplier = Chem.SDMolSupplier(file_path)
        if len(supplier) > 0:
            mol = supplier[0]

    elif file_extension == ".pdb":
        # For PDB files
        mol = Chem.MolFromPDBFile(file_path)

    elif file_extension in [".mol2", ".ml2"]:
        # For MOL2 files (requires rdkit.Chem.AllChem)
        mol = Chem.MolFromMol2File(file_path)

    elif file_extension == ".xyz":
        # For XYZ files (basic implementation)
        mol = Chem.MolFromXYZFile(file_path)

    elif file_extension == ".inchi":
        # For InChI files
        with open(file_path, "r") as f:
            inchi = f.readline().strip()
            mol = Chem.MolFromInchi(inchi)

    if mol is None:
        raise ValueError(f"Failed to load molecule from {file_path}")

    return mol


def predict_bonds(model: torch.nn.Module, mol: Chem.Mol) -> Chem.Mol:
    """Predict bonds for a molecule."""

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
            raise ValueError("Invalid bond type")

        mol.AddBond(int(i), int(j), bond_order)

    return mol


def save_molecule(mol: Chem.Mol, file_path: str):
    """
    Save a molecule to a file using RDKit based on the file extension.

    Parameters:
    mol (Chem.Mol): RDKit molecule object to save
    file_path (str): Path where the molecule should be saved

    Returns:
    bool: True if successful, False otherwise
    """
    if mol is None:
        print("Error: No molecule to save")
        return False

    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Handle different file formats based on extension
    if file_extension in [".smi", ".smiles"]:
        # Save as SMILES
        smiles = Chem.MolToSmiles(mol)
        with open(file_path, "w") as f:
            f.write(smiles)

    elif file_extension == ".mol":
        # Save as MDL MOL file
        return Chem.MolToMolFile(mol, file_path)

    elif file_extension == ".sdf":
        # Save as SDF file
        writer = Chem.SDWriter(file_path)
        writer.write(mol)
        writer.close()

    elif file_extension == ".pdb":
        # Save as PDB file
        return Chem.MolToPDBFile(mol, file_path)

    elif file_extension == ".inchi":
        # Save as InChI
        inchi = Chem.MolToInchi(mol)
        with open(file_path, "w") as f:
            f.write(inchi)

    elif file_extension in [".png", ".jpg", ".jpeg"]:
        # Save as image
        # Generate 2D coordinates if they don't exist
        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)
        MolToFile(mol, file_path)

    elif file_extension == ".svg":
        # Save as SVG
        # Generate 2D coordinates if they don't exist
        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)
        MolToFile(mol, file_path)

    else:
        raise ValueError(f"Unsupported file extension for saving: {file_extension}")


def load_model(exported_model_path: str):
    """Load the exported model."""
    # Check if relative path is provided.
    if not os.path.isabs(exported_model_path):
        current_path = Path(__file__).resolve()
        project_root = current_path.parent.parent.parent
        exported_model_path = os.path.join(project_root, exported_model_path)
    model = torch._inductor.aoti_load_package(exported_model_path)
    return model


def predict_bonds_and_save(input_file: str, exported_model_path: str, output_file: str, output_extension: str = None):
    """Load a molecule, predict bonds, and save the molecule with predicted bonds."""

    # Load model.
    model = load_model(exported_model_path)

    def run_on_file(input_file: str, output_file: str):
        # Load molecule.
        mol = load_molecule(input_file)

        # Predict bonds.
        mol_with_bonds = predict_bonds(model, mol)

        # Save molecule with bonds.
        save_molecule(mol_with_bonds, output_file)

    if os.path.isdir(input_file):
        os.makedirs(output_file, exist_ok=True)
        assert os.path.isdir(output_file), "Output must be a directory if input is a directory"
        assert output_extension is not None, "Output extension must be provided if input is a directory"
    else:
        if os.path.isdir(output_file):
            assert output_extension is not None, "Output extension must be provided if output is a directory"
            output_file = os.path.join(
                output_file, os.path.basename(input_file)[: -len(output_extension)] + output_extension
            )

    # If input is a directory, process all files in the directory
    count = 0
    success = 0
    if os.path.isdir(input_file):
        for file_name in tqdm.tqdm(os.listdir(input_file)):
            file_path = os.path.join(input_file, file_name)
            if os.path.isfile(file_path):
                count += 1
                output_path = os.path.join(output_file, file_name[: -len(output_extension)] + output_extension)
                try:
                    run_on_file(file_path, output_path)
                    success += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    else:
        count = 1
        run_on_file(input_file, output_file)
        success = 1

    print(f"Processed {success} out of {count} files: {success / count * 100:.2f}% success rate")


def main():
    parser = argparse.ArgumentParser(description="Predict bond type")
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument(
        "--exported_model_path", type=str, help="Path to the exported model", default="exported/bond_predictor.pt2"
    )
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("--extension", type=str, help="Output file extension", default=None)
    args = parser.parse_args()

    if args.extension is not None:
        if not args.extension.startswith("."):
            args.extension = "." + args.extension

        if args.extension not in [".smi", ".smiles", ".mol", ".sdf", ".pdb", ".inchi", ".png", ".jpg", ".jpeg", ".svg"]:
            raise ValueError(f"Unsupported output extension: {args.extension}")

    predict_bonds_and_save(args.input, args.exported_model_path, args.output, args.extension)
