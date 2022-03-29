# Source: https://github.com/krrish94/nerf-pytorch

# Torch imports
import torch
import pytorch_lightning as pl
from torch.nn import functional as F

# Other imports
import numpy as np

# File imports
import utils


class LightningModule(pl.LightningModule):
    def __init__(self, model, 
                 image_size=800,
                 is_debug=False,
                 num_encoders=6, 
                 num_viewdir_encoders=0,
                 num_ray_samples=32, 
                 lr=5e-3,
                 log_every=100):
        super().__init__()
        
        self.model = model
        
        self.image_size = image_size
        self.num_encoders = num_encoders
        self.num_viewdir_encoders = num_viewdir_encoders
        self.num_ray_samples = num_ray_samples
        self.lr = lr
        self.log_every = log_every
        self.is_debug = is_debug
        
        self.focal_length = 138.8889 if is_debug else 875 / (800 / image_size)
        self.near_threshold = 2. if is_debug else 1.0
        self.far_threshold = 6. if is_debug else 5.
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def step(self, batch, split):
        name, rgb, pose = batch
        name = name[0]
        rgb = rgb.squeeze()
        pose = pose.squeeze()
        
        # Get the "bundle" of rays through all image pixels.
        ray_origins, ray_directions = utils.get_ray_bundle(self.image_size, self.image_size, self.focal_length, pose)
        
        # Sample points along each ray
        ray_samples, depth_values = utils.sample_rays(ray_origins, ray_directions, self.near_threshold, self.far_threshold, self.num_ray_samples)
        
        # Add positional encoding
        encoded_points = utils.positional_encoding(ray_samples, num_encoders=self.num_encoders)
        
        if self.num_viewdir_encoders > 0:
            viewdirs = ray_directions / ray_directions.norm(p=2, dim=-1).unsqueeze(-1)
            viewdirs = viewdirs.reshape((-1, 3)).unsqueeze(1)

            input_dirs = viewdirs.expand((self.image_size*self.image_size, self.num_ray_samples, 3))
            input_dirs = input_dirs.reshape((-1, input_dirs.shape[-1]))
            
            embedded_dirs = utils.positional_encoding(input_dirs, num_encoders=self.num_viewdir_encoders)
            encoded_points = torch.cat((encoded_points, embedded_dirs), dim=-1)

        # Calculate radiance field using model
        radiance_field = self.model(encoded_points).reshape(*ray_samples.shape[:-1], 4)
        # Perform differentiable volume rendering to re-synthesize the RGB image
        output, depth_map, acc_map = utils.render_volume_density(radiance_field, ray_origins, depth_values, not self.is_debug)
          
        # Calculate loss & PSNR
        loss = torch.nn.functional.mse_loss(output, rgb)
        self.log(split+"loss", loss, on_step=True, on_epoch=True, logger=True)
        psnr = -10. * torch.log10(loss)
        self.log(split+"psnr", psnr, on_step=True, on_epoch=True, logger=True)
        
        # Log visualization every X epochs
        if self.logger is not None and split == 'val/' and self.current_epoch % (self.log_every / 100) == 0:
            self.logger.experiment.add_image("Epoch " + str(self.current_epoch), torch.permute(output, (2,0,1)), self.current_epoch)
        
        # Log test RGB & depth images
        if self.logger is not None and split == 'test/':
            self.logger.experiment.add_image("RGB_" + name, torch.permute(output, (2,0,1)))
            self.logger.experiment.add_image("DEPTH_" + name, depth_map.unsqueeze(0))
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, 'train/')
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, 'val/')
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch, 'test/')
