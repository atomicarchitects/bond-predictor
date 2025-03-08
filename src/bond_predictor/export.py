import os
from pathlib import Path

import torch
import torch.export
import torch.nn as nn
import argparse

import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
torch.set_float32_matmul_precision("high")

import e3tools
import rdkit
import rdkit.Chem as Chem

from bond_predictor import models, rdkit_utils


class ModelWrapper(nn.Module):
    """Wrapper for export."""

    def __init__(self, net, input_keys, output_keys):
        super(ModelWrapper, self).__init__()
        self.net = net
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, coordinates, atom_types, edge_index):
        return self.net(coordinates, atom_types, edge_index)


def export_model(weights_path: str, output_path: str) -> str:
    """Export the model to a AOTI package, and return the path to the package."""
    net = models.E3Conv()
    net = torch.compile(net, dynamic=True, fullgraph=True)

    weights = torch.load(weights_path)
    net.load_state_dict(weights)

    output_keys = ["bond_logits", "charge_logits"]
    input_keys = ["coordinates", "atom_types", "edge_index"]

    wrapped_net = ModelWrapper(net, input_keys, output_keys)
    dummy_data = (torch.randn(10, 3), torch.randint(0, 100, (10,)), torch.randint(0, 10, (2, 15)))

    num_nodes = torch.export.Dim("num_nodes", min=0, max=1000000)
    num_edges = torch.export.Dim("num_edges", min=0, max=1000000)
    dynamic_shapes = {"coordinates": {0: num_nodes}, "atom_types": {0: num_nodes}, "edge_index": {1: num_edges}}

    exported = torch.export.export(
        wrapped_net,
        dummy_data,
        dynamic_shapes=dynamic_shapes,
    )

    # Save the exported model to a file.
    # Load with torch._inductor.aoti_load_package(out_path).
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent.parent
    output_path = os.path.join(project_root, output_path)
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
    )

    print(f"Model exported to {os.path.abspath(output_path)}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=str, help="Path to the weights.")
    parser.add_argument("--output", type=str, help="Path to the output package.", default="exported/bond_predictor.pt2")
    args = parser.parse_args()

    export_model(args.weights, args.output)


if __name__ == "__main__":
    main()
