import torch, torch.nn as nn, torch.optim as optim
from benchmark.template_torch_cv import train_and_eval

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

import yaml
with open("../config/mlp.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = MLP(input_dim=22, hidden_dim=cfg["model"]["hidden_dim"])  # x(1) + feature(21)
criterion = nn.MSELoss()
optimizer = optim.Adam
train_and_eval("../config/mlp.yaml", model, "mlp", criterion, optimizer, lr=cfg["model"]["lr"], epochs=cfg["training"]["epochs"])