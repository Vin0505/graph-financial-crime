
import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

# Use the GPU if available; otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Synthetic graph: 6 nodes, 4 features per node.
x = torch.randn(6, 4, device=device)

# Each column represents a directed edge: source -> target.
edge_index = torch.tensor(
    [
        [0, 1, 2, 3, 4, 5, 0, 2],
        [1, 2, 3, 4, 5, 0, 2, 4],
    ],
    dtype=torch.long,
    device=device,
)

# Labels for four nodes; the other two are not used in the loss.
y = torch.tensor([0, 1, 0, 1], device=device)


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 8)
        self.conv2 = GCNConv(8, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(4, 8)
        self.conv2 = SAGEConv(8, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


for name, model in [
    ("GCN", GCN()),
    ("GraphSAGE", GraphSAGE()),
]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    optimizer.zero_grad()

    logits = model(x, edge_index)
    loss = nn.functional.cross_entropy(logits[:4], y)

    loss.backward()

    gradients_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    assert gradients_ok, f"{name}: invalid or missing gradients"

    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"{name}: device={device}, "
        f"logits_shape={tuple(logits.shape)}, "
        f"loss={loss.item():.4f}, "
        f"gradients_ok={gradients_ok}"
    )

print("GNN smoke test passed.")
