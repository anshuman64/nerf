# Source: https://github.com/krrish94/nerf-pytorch

# Torch imports
import torch
from torch.nn import functional as F

# Other imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Optional


#####################
## Raymarching
#####################

def get_ray_bundle(height, width, focal_length, tform_cam2world):
    ii, jj = torch.meshgrid(torch.arange(width).to(tform_cam2world),
                            torch.arange(height).to(tform_cam2world), 
                            indexing='xy')
    
    directions = torch.stack([(ii - width * .5) / focal_length,
                            -(jj - height * .5) / focal_length,
                            -torch.ones_like(ii)], dim=-1)
    
    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    
    return ray_origins, ray_directions

def sample_rays(ray_origins, 
               ray_directions, 
               near_threshold, 
               far_threshold, 
               num_samples):
    
    depth_values = torch.linspace(near_threshold, far_threshold, num_samples, device=ray_origins.device)
    ray_samples = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * depth_values.unsqueeze(1)
        
    return ray_samples, depth_values


#####################
## Volumetric Rendering
#####################

def render_volume_density(radiance_field, ray_origins, depth_values, white_background = True):
    rgb = torch.sigmoid(radiance_field[..., :3])
    sigma = torch.nn.functional.relu(radiance_field[..., 3])
    
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    one_e_10 = one_e_10.expand(depth_values[..., :1].shape)
    
    dists = depth_values[..., 1:] - depth_values[..., :-1]
    dists = torch.cat((dists, one_e_10), dim=-1)
    
    alpha = 1. - torch.exp(-sigma * dists)
    cumprod = torch.cumprod(1. - alpha + 1e-10, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    weights = alpha * cumprod

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)
    
    if white_background:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map


def positional_encoding(tensor, num_encoders=6):
    frequency_bands = 2.0 ** torch.linspace(0.0,
        num_encoders - 1,
        num_encoders,
        dtype=tensor.dtype,
        device=tensor.device)

    tensor = tensor.reshape((-1, tensor.shape[-1]))
    encoding = [tensor]
    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    return torch.cat(encoding, dim=-1)

