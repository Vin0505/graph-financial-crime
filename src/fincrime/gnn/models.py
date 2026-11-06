import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def build_gnn_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int = 32,
    dropout: float = 0.3,
    num_classes: int = 2,
) -> nn.Module:
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")

    if hidden_channels <= 0:
        raise ValueError("hidden_channels must be positive.")

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1).")

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    model_name = model_name.lower()

    model_classes = {
        "gcn": GCN,
        "graphsage": GraphSAGE,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unsupported GNN model: {model_name}")

    return model_classes[model_name](
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dropout=dropout,
        num_classes=num_classes,
    )
