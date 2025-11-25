from src.feature_lists import ALL_FEATURES

import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TwoTowerSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int = len(ALL_FEATURES),
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.3,
        embed_l2_reg: float = 1e-4,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.sage1_conv1 = SAGEConv(in_dim, hidden_dim)
        self.sage1_conv2 = SAGEConv(hidden_dim, out_dim)

        self.sage2_conv1 = SAGEConv(in_dim, hidden_dim)
        self.sage2_conv2 = SAGEConv(hidden_dim, out_dim)
        
        # ADD: MLP prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out_dim * 2 + 1, hidden_dim),  # Concatenate e1, e2, src_pct
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = dropout
        self.embed_l2_reg = embed_l2_reg
        self.normalize_embeddings = normalize_embeddings

    def encode_e1(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x1 = self.sage1_conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        e1 = self.sage1_conv2(x1, edge_index)
        return e1

    def encode_e2(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x2 = self.sage2_conv1(x, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        e2 = self.sage2_conv2(x2, edge_index)
        return e2

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        pct_change: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, F]
        edge_index: [2, E]
        src_idx, tgt_idx: [B] long
        pct_change: [N] float
        returns: y_hat [B], E1 [N,D], E2 [N,D]
        """
        E1 = self.encode_e1(x, edge_index)  # [N,D]
        E2 = self.encode_e2(x, edge_index)  # [N,D]

        # Normalize embeddings for stability
        if self.normalize_embeddings:
            E1 = F.normalize(E1, p=2, dim=-1)
            E2 = F.normalize(E2, p=2, dim=-1)

        # Gather pairs
        e1_src = E1[src_idx]  # [B,D]
        e2_tgt = E2[tgt_idx]  # [B,D]
        src_pct = pct_change[src_idx].unsqueeze(1)  # [B, 1]

        # Concatenate embeddings and source percentage
        combined = torch.cat([e1_src, e2_tgt, src_pct], dim=-1)  # [B, D*2+1]

        # Pass through MLP for prediction
        y_hat = self.predictor(combined).squeeze(-1)  # [B, 1] -> [B]

        return y_hat, E1, E2

    def embedding_regularization(self, E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
        """Return L2 regularization term for embeddings."""
        if self.embed_l2_reg <= 0 or E1.shape[0] == 0:
            return torch.tensor(0.0, device=E1.device)
        # Use mean to be independent of graph size
        reg = (E1.pow(2).mean() + E2.pow(2).mean()) * self.embed_l2_reg
        return reg