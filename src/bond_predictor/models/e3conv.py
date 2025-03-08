from typing import Callable, Optional, Union
import functools

import e3nn
import torch
import torch.nn as nn
from e3nn import o3
import e3tools
import e3tools.nn


class E3Conv(nn.Module):
    """A simple E(3)-equivariant convolutional neural network, similar to NequIP."""

    def __init__(
        self,
        irreps_hidden: str = "32x0e + 32x1o",
        irreps_sh: str = "1x0e + 1x1o",
        num_layers: int = 1,
        num_bond_types: int = 5,
        num_atom_types: int = 100,
        max_charge: int = 6,
        edge_attr_dim: int = 8,
        radial_cutoff: float = 5.0,  # Angstrom
    ):
        super().__init__()

        self.irreps_out = o3.Irreps(f"{num_bond_types}x0e")
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_layers = num_layers
        self.edge_attr_dim = edge_attr_dim
        self.radial_cutoff = radial_cutoff
        self.max_charge = max_charge

        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, normalize=True, normalization="component")
        self.atom_embedder = nn.Embedding(num_atom_types, 32)

        self.initial_projector = e3tools.nn.Conv(
            irreps_in=f"{self.atom_embedder.embedding_dim}x0e",
            irreps_out=self.irreps_hidden,
            irreps_sh=self.irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                e3tools.nn.LinearSelfInteraction(
                    e3tools.nn.Conv(
                        irreps_in=self.irreps_hidden,
                        irreps_out=self.irreps_hidden,
                        irreps_sh=self.irreps_sh,
                        edge_attr_dim=self.edge_attr_dim,
                    )
                )
            )

        self.edge_tp = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_hidden + self.irreps_hidden,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            shared_weights=False,
            internal_weights=False,
        )
        self.edge_tp_radial_nn = e3tools.nn.ScalarMLP(
            edge_attr_dim,
            self.edge_tp.weight_numel,
            hidden_features=[edge_attr_dim],
            activation_layer=torch.nn.SiLU,
        )

        self.node_output = e3tools.nn.EquivariantMLP(
            irreps_in=self.irreps_hidden,
            irreps_out=f"{max_charge * 2 + 1}x0e",
            irreps_hidden_list=[f"{max_charge * 2 + 1}x0e"],
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = e3tools.radius_graph(coordinates, r=self.radial_cutoff)
        edge_vec = coordinates[edge_index[1]] - coordinates[edge_index[0]]
        edge_sh = self.sh(edge_vec)

        edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            self.radial_cutoff,
            self.edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )

        node_attr = self.atom_embedder(atom_types)
        node_attr = self.initial_projector(node_attr, edge_index, edge_attr, edge_sh)
        for layer in self.layers:
            node_attr = layer(node_attr, edge_index, edge_attr, edge_sh)

        # Compute edge features.
        edge_attr_final = torch.cat([node_attr[edge_index[0]], node_attr[edge_index[1]]], dim=-1)
        edge_attr_final = self.edge_tp(edge_attr_final, edge_sh, self.edge_tp_radial_nn(edge_attr))

        # Compute node features.
        node_attr = node_attr.detach()
        node_attr_final = self.node_output(node_attr)

        return edge_attr_final, node_attr_final

        # # Map edge features to bond logits.
        # N = coordinates.shape[0]
        # bond_logits = torch.zeros((N, N, edge_attr_final.shape[-1]), dtype=edge_attr_final.dtype, device=edge_attr_final.device)

        # # For the non-edges, set the bond type to "no bond".
        # bond_logits[:, :, 0] = 1e6  # No bond
        # bond_logits[edge_index[0], edge_index[1]] = edge_attr_final
        # bond_logits[edge_index[1], edge_index[0]] = edge_attr_final

        # return bond_logits
