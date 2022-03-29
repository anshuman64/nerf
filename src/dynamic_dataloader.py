# Torch imports
import torch
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Other package imports
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

# File imports
import utils

DATA_PATH = "/userdata/kerasData/old/anshuman-test/su/HW3/data/"

class DynamicDataModule(pl.LightningDataModule):
    def __init__(self, is_debug=False, image_size=800):
        super().__init__()
        
        self.is_debug = is_debug
        self.image_size = image_size
        self.has_setup = False
        
    def setup(self, stage=None):
        if self.has_setup: return
    
        if self.is_debug:
            data = np.load(DATA_PATH + "tiny_nerf_data.npz")
            images = data["images"]
            poses = data["poses"]
            
            # Use 5 images for test
            self.test_data = {"images": images[-5:], "poses": poses[-5:]}
            # Split rest for train/val
            train_images, val_images, train_poses, val_poses = train_test_split(images[:-5], poses[:-5], test_size=0.2)
            self.train_data = {"images": train_images, "poses": train_poses}
            self.val_data = {"images": val_images, "poses": val_poses}
            
        else:
            file_list = np.loadtxt(DATA_PATH + "images.txt", dtype=str)

            self.train_data, self.val_data = train_test_split(file_list, test_size=0.2)
            self.test_data = ["2_test_0000", "2_test_0016", "2_test_0055", "2_test_0093", "2_test_0160"]
        
        self.has_setup = True
                
    def train_dataloader(self):
        train_dataloader = DynamicDataLoader(self.train_data, 
                                             is_debug=self.is_debug,
                                             image_size=self.image_size)
        
        return DataLoader(train_dataloader, 
                             batch_size=1,
                             num_workers=4,
                             shuffle=True,
                             pin_memory=True)
    
    def val_dataloader(self):
        val_dataloader = DynamicDataLoader(self.val_data, 
                                             is_debug=self.is_debug,
                                             image_size=self.image_size)
        
        return DataLoader(val_dataloader, 
                             batch_size=1,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=True)
    
    def test_dataloader(self):
        test_dataloader = DynamicDataLoader(self.test_data, 
                                             is_debug=self.is_debug,
                                             image_size=self.image_size)
        
        return DataLoader(test_dataloader, 
                             batch_size=1,
                             num_workers=0,
                             shuffle=False,
                             pin_memory=True)
    

#####################
## Dataloader
#####################
    
class DynamicDataLoader(Dataset):
    def __init__(self, data, is_debug=False, image_size=800):
        self.data = data
        self.is_debug = is_debug
        self.image_size = image_size

    def __len__(self):
        if self.is_debug:
            return len(self.data["images"])
            
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_debug:
            # Lego dataset
            name = ""
            rgb = self.data["images"][idx]
            pose = self.data["poses"][idx]
        else:
            # Bottles dataset
            name = self.data[idx]
            
            try:
                # If image exists, load it
                rgb = Image.open(DATA_PATH + "rgb/" + name + ".png").convert("RGB")
            except:
                # Else, is test image and load zeros
                rgb = np.zeros((self.image_size, self.image_size, 3))
            
            rgb = np.array(rgb)/255
            rgb = cv2.resize(rgb, dsize=(self.image_size, self.image_size))

            pose = np.loadtxt(DATA_PATH + "pose/" + name + ".txt", dtype=float)
            pose[:, 1:3] *= -1
        
        return name, rgb, pose
