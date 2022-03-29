# Source: https://github.com/krrish94/nerf-pytorch

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F


class VeryTinyNerfModel(torch.nn.Module):
    def __init__(self, hidden_size=128, num_encoders=6):
        super(VeryTinyNerfModel, self).__init__()
        
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoders, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 4)
  
    def forward(self, x):
        x = x.float()
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x
    

class ReplicateNeRFModel(torch.nn.Module):
    def __init__(self,
        hidden_size=256,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4):
        super(ReplicateNeRFModel, self).__init__()
        
        self.dim_xyz = 3 + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if num_encoding_fn_dir > 0 else 0) + 2 * 3 * max(num_encoding_fn_dir, 0)

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.rgb = torch.nn.Linear(hidden_size // 2, 3)

    def forward(self, x):
        x = x.float()
        xyz, direction = x[...,:self.dim_xyz], x[...,self.dim_xyz:]
        
        # Pass only location first
        x = F.relu(self.layer1(xyz))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        alpha = self.alpha(x)
        
        # Add viewing direction
        x = F.relu(self.layer4(torch.cat((x, direction), dim=-1)))
        x = F.relu(self.layer5(x))
        rgb = self.rgb(x)
        
        return torch.cat((rgb, alpha), dim=-1)