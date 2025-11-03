import torch, torch.nn as nn, torch.optim as optim, yaml
from benchmark.template_torch_cv import train_and_eval

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x).mean(dim=1)
        return self.fc(x)

with open("../config/transformer.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = TransformerRegressor(
    input_dim=22,
    d_model=cfg["model"]["d_model"],
    nhead=cfg["model"]["nhead"],
    num_layers=cfg["model"]["num_layers"]
)

criterion = nn.MSELoss()
optimizer = optim.Adam
train_and_eval("../config/transformer.yaml", model, "transformer", criterion, optimizer, lr=cfg["model"]["lr"], epochs=cfg["training"]["epochs"])