from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GENConv,
    LayerNorm,
    PDNConv,
    PNAConv,
    TransformerConv,
)
from torch_geometric.utils import dropout_edge


class CDEmbedder(nn.Module):
    def __init__(self, output_dim: int):
        super(CDEmbedder, self).__init__()
        self.emb_wid = nn.Embedding(13, 16)
        self.emb_ken = nn.Embedding(48, 16)
        self.emb_lrg = nn.Embedding(300, 32)
        self.emb_sml = nn.Embedding(538, 32)
        self.linear = nn.Linear(16 + 16 + 32 + 32, output_dim)

    def forward(self, x):
        win = self.emb_wid(x[:, 0].long())
        ken = self.emb_ken(x[:, 1].long())
        lrg = self.emb_lrg(x[:, 2].long())
        sml = self.emb_sml(x[:, 3].long())
        x = torch.cat([win, ken, lrg, sml], dim=1)
        return self.linear(x)


def get_conv(
    conv_type: str,
    in_ch: int,
    out_ch: int,
    edge_dim: int,
    deg: Optional[torch.Tensor] = None,
    conv_params: dict = {},
):
    if conv_type == "gat":
        if "heads" in conv_params:
            out_ch = out_ch // conv_params["heads"]
        return GATv2Conv(in_ch, out_ch, edge_dim=edge_dim, **conv_params)
    elif conv_type == "transformer":
        if "heads" in conv_params:
            out_ch = out_ch // conv_params["heads"]
        return TransformerConv(
            in_channels=in_ch,
            out_channels=out_ch,
            edge_dim=edge_dim,
            **conv_params,
        )
    elif conv_type == "pna":
        conv_params["aggregators"] = list(conv_params["aggregators"])
        conv_params["scalers"] = list(conv_params["scalers"])
        return PNAConv(
            in_channels=in_ch,
            out_channels=out_ch,
            deg=deg,
            edge_dim=edge_dim,
            **conv_params,
        )
    elif conv_type == "gen":
        return GENConv(in_channels=in_ch, out_channels=out_ch, edge_dim=edge_dim, **conv_params)
    elif conv_type == "pdn":
        return PDNConv(in_channels=in_ch, out_channels=out_ch, edge_dim=edge_dim, **conv_params)
    else:
        raise ValueError(f"conv_type {conv_type} is not supported, use gat, transformer, pna")


def get_norm(norm_type: str, mid_dim: int):
    if norm_type == "batch":
        return BatchNorm(mid_dim)
    elif norm_type == "layer":
        return LayerNorm(mid_dim)
    else:
        raise ValueError(f"norm_type {norm_type} is not supported, use batch or layer")


class YadGNN(nn.Module):
    """ResGCN+ with cd embedding
    参考: https://github.com/knshnb/kaggle-tpu-graph-5th-place/blob/master/src/nn.py

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_layers: int = 3,
        mid_dim: int = 256,
        dropout_rate: float = 0.2,
        edge_dropout_rate: float = 0.1,
        norm_type: str = "layer",
        conv_type: str = "gat",
        conv_params: dict = {},
        deg: Optional[torch.Tensor] = None,
    ):
        super(YadGNN, self).__init__()

        assert conv_type != "pna" or deg is not None, "deg must be provided for pna"

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_layers = num_layers
        self.mid_dim = mid_dim
        self.dropout_rate = dropout_rate
        self.edge_dropout_rate = edge_dropout_rate

        # cd embedding
        self.cd_emb = CDEmbedder(256)
        self.linear1 = nn.Linear(256 + num_node_features, mid_dim)

        # graph conv
        self.convs = ModuleList()
        self.rev_convs = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            conv = get_conv(
                conv_type,
                mid_dim,
                mid_dim // 2,
                edge_dim=num_edge_features,
                conv_params=conv_params,
                deg=deg,
            )
            self.convs.append(conv)
            conv = get_conv(
                conv_type,
                mid_dim,
                mid_dim // 2,
                edge_dim=num_edge_features,
                conv_params=conv_params,
                deg=deg,
            )
            self.rev_convs.append(conv)
            norm = get_norm(norm_type, mid_dim)
            self.norms.append(norm)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear2 = nn.Linear(mid_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # dropout edge
        edge_index, edge_mask = dropout_edge(
            edge_index, p=self.edge_dropout_rate, training=self.training
        )
        edge_attr = edge_attr[edge_mask]
        rev_edge_index = torch.flip(edge_index, dims=(0,))  # reverse edge

        # pre linear
        cd_emb = self.cd_emb(x[:, :4])
        x = torch.cat([cd_emb, x[:, 4:]], dim=1)
        x = self.linear1(x)

        # ResGCN+
        x_prev = x
        for i in range(len(self.convs)):
            # normalize
            x = self.norms[i](x)
            # relu
            x = F.relu(x)
            # conv
            x_1 = self.convs[i](x, edge_index, edge_attr)
            x_2 = self.rev_convs[i](x, rev_edge_index, edge_attr)
            x = torch.cat([x_1, x_2], dim=1)
            # add residual
            x = x + x_prev
            x_prev = x

        # post linear
        x = self.dropout(x)
        x = self.linear2(x).flatten()

        return x
