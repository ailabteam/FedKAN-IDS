import torch
import torch.nn as nn
from src.models.kan import KAN

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

input_dim = 43
output_dim = 2

# Cấu hình hiện tại
kan_model = KAN([43, 8, 2], grid_size=5, spline_order=3) # Hidden 8
mlp_model = MLP(43, 64, 2) # Hidden 64

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

kan_params = count_parameters(kan_model)
mlp_params = count_parameters(mlp_model)

print(f"KAN Parameters: {kan_params:,}")
print(f"MLP Parameters: {mlp_params:,}")
print(f"Ratio (MLP/KAN): {mlp_params/kan_params:.2f}x")

