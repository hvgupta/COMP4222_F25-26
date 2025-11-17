from src.feature_lists import ALL_FEATURES

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

NUM_FEATURES = len(ALL_FEATURES) - 2 # this removes dates and features

class TwoTowerSAGE(nn.Module):
    """
    Includes two models, 
        one is responsible for generating an embedding which can be combined with the pct to get the "source embedding"
        the other is the target embedding.
        
        both are combined together (using dot product for now) 
        the objective is to minimize the difference between the dot product and the actual pct change of the trget stock
    """
    def __init__(self, in_dim = len(ALL_FEATURES), hidden_dim=64, out_dim=32, dropout=0.2):
        super().__init__()
        
        self.sage1_conv1 = SAGEConv(in_dim, hidden_dim)
        self.sage1_conv2 = SAGEConv(hidden_dim, out_dim)
        
        self.sage2_conv1 = SAGEConv(in_dim, hidden_dim)
        self.sage2_conv2 = SAGEConv(hidden_dim, out_dim)

        self.dropout = dropout

    def encode_e1(self, x, edge_index):
        x1 = F.relu(self.sage1_conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        e1 = self.sage1_conv2(x1, edge_index)
        return e1  # shape [num_nodes, out_dim]

    def encode_e2(self, x, edge_index):
        x2 = F.relu(self.sage2_conv1(x, edge_index))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        e2 = self.sage2_conv2(x2, edge_index)
        return e2  # shape [num_nodes, out_dim]

    def forward(self, x, edge_index, src_idx, tgt_idx, src_pct):
        """
        x: [num_nodes, in_dim] node features tensor
        edge_index: [2, num_edges]
        src_idx: [batch_size] tensor of source node indices
        tgt_idx: [batch_size] tensor of target node indices
        src_pct: [batch_size] tensor of today's pct change for source nodes (float)
        """
        # produce both embeddings
        E1 = self.encode_e1(x, edge_index)  # [N, D]
        E2 = self.encode_e2(x, edge_index)  # [N, D]

        # gather embeddings for pairs
        e1_src = E1[src_idx]   # [B, D]
        e2_tgt = E2[tgt_idx]   # [B, D]

        # apply conditioning by pct: broadcast scalar to D-dim
        # if you prefer percent as fraction (-0.02 etc)
        src_pct = src_pct.view(-1, 1)     # [B, 1]
        cond = e1_src * src_pct           # [B, D]

        # predicted scalar = dot(cond, e2_tgt) per example
        y_hat = (cond * e2_tgt).sum(dim=-1)   # [B]

        return y_hat, E1, E2