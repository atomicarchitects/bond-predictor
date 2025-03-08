import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """A 3D point cloud bond type predictor using the Transformer architecture."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        num_bond_types: int = 5,
        num_atom_types: int = 100,
    ):
        """
        Point Cloud Bond Type Predictor using Transformer architecture

        Args:
            hidden_dim: Hidden dimension size for the transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_bond_types: Number of different bond types to predict (including no bond)
            num_atom_types: Number of different atom types in the point cloud
        """
        super().__init__()

        # Atom type embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim // 2)

        # Point coordinate embedding (3D coordinates → hidden_dim//2)
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 4), nn.SELU(), nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        # Fusion layer to combine atom type and coordinate embeddings
        self.fusion_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SELU())

        # Transformer layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)])

        # Bond type prediction layers
        self.bond_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SELU(), nn.Linear(hidden_dim, num_bond_types)
        )

        self.hidden_dim = hidden_dim
        self.num_bond_types = num_bond_types

    def forward(self, coordinates: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coordinates: Point cloud tensor of shape (batch_size, num_points, 3)
            atom_types: Atom type indices of shape (batch_size, num_points)

        Returns:
            bond_logits: Bond type logits of shape (batch_size, num_points, num_points, num_bond_types)
        """
        if coordinates.dim() != 3:
            coordinates = coordinates[None, ...]

        if atom_types.dim() != 2:
            atom_types = atom_types[None, ...]

        batch_size, num_points, _ = coordinates.shape

        # Embed atom types
        atom_features = self.atom_embedding(atom_types)  # Shape: (batch_size, num_points, hidden_dim//2)

        # Embed coordinates
        coord_features = self.coord_embedding(coordinates)  # Shape: (batch_size, num_points, hidden_dim//2)

        # Concatenate and fuse features
        combined_features = torch.cat([coord_features, atom_features], dim=-1)
        point_features = self.fusion_layer(combined_features)  # Shape: (batch_size, num_points, hidden_dim)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            point_features = transformer_layer(point_features)

        # Create all pairs of point features for bond prediction
        # First, create tensors that will be used to create all pairs
        point_features_i = point_features.unsqueeze(2).expand(batch_size, num_points, num_points, self.hidden_dim)
        point_features_j = point_features.unsqueeze(1).expand(batch_size, num_points, num_points, self.hidden_dim)

        # Concatenate features for each pair of points
        pair_features = torch.cat([point_features_i, point_features_j], dim=-1)

        # Predict bond types for each pair
        bond_logits = self.bond_predictor(pair_features)

        return bond_logits


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        # Multi-head self-attention using PyTorch's implementation
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.SELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
