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

        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = dropout
        self.embed_l2_reg = embed_l2_reg
        self.normalize_embeddings = normalize_embeddings

    def encode_e1(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.sage1_conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        e1 = self.sage1_conv2(x1, edge_index)
        e1 = self.bn1(e1)
        return e1  # [N, D]

    def encode_e2(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x2 = F.relu(self.sage2_conv1(x, edge_index))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        e2 = self.sage2_conv2(x2, edge_index)
        e2 = self.bn2(e2)
        return e2  # [N, D]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        src_pct: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, F]
        edge_index: [2, E]
        src_idx, tgt_idx: [B] long
        src_pct: [B] float (prefer fraction, e.g. 0.012 for +1.2%)
        returns: y_hat [B], E1 [N,D], E2 [N,D]
        """
        E1 = self.encode_e1(x, edge_index)  # [N,D]
        E2 = self.encode_e2(x, edge_index)  # [N,D]

        # Optionally normalize embeddings onto unit sphere for stability (cosine-like)
        if self.normalize_embeddings:
            E1 = F.normalize(E1, p=2, dim=-1)
            E2 = F.normalize(E2, p=2, dim=-1)

        # gather pairs
        e1_src = E1[src_idx]  # [B,D]
        e2_tgt = E2[tgt_idx]  # [B,D]

        # condition e1 by scalar src_pct: we scale & optionally clamp src_pct beforehand
        # Broadcast to D
        src_pct = src_pct.view(-1, 1)  # [B,1]
        cond = e1_src * src_pct  # [B,D]

        # raw dot
        y_hat = (cond * e2_tgt).sum(dim=-1)  # [B]

        return y_hat, E1, E2

    def embedding_regularization(self, E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
        """Return L2 regularization term for embeddings (scalar)."""
        if self.embed_l2_reg <= 0:
            return torch.tensor(0.0, device=E1.device)
        reg = (E1.norm(p=2) ** 2 + E2.norm(p=2) ** 2) * (self.embed_l2_reg / (E1.shape[0]))
        return reg