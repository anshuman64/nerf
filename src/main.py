# Torch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# File imports
from lightning_module import LightningModule
from main_model import *
from dynamic_dataloader import DynamicDataModule


##############
## Constants
###############

IS_DEBUG = False
IMAGE_SIZE = 200
NUM_ENCODERS = 6
NUM_VIEWDIR_ENCODERS = 4
NUM_RAY_SAMPLES = 32
LR = 5e-4
NUM_EPOCHS = 501
LOG_EVERY = 10000
CHECKPOINT_PATH = None # '/userdata/kerasData/old/anshuman-test/su/HW3/lightning_logs/default/version_2/checkpoints/epoch=99-step=15999.ckpt'



data_module = DynamicDataModule(IS_DEBUG, IMAGE_SIZE)

main_model = ReplicateNeRFModel(num_encoding_fn_xyz=NUM_ENCODERS, num_encoding_fn_dir=NUM_VIEWDIR_ENCODERS)
# main_model = VeryTinyNerfModel(num_encoders=NUM_ENCODERS)

if CHECKPOINT_PATH is not None:
    lightning_module = LightningModule.load_from_checkpoint(CHECKPOINT_PATH,
                                   model=main_model,
                                   image_size=IMAGE_SIZE,
                                   is_debug=IS_DEBUG,
                                   num_encoders=NUM_ENCODERS, 
                                   num_viewdir_encoders=NUM_VIEWDIR_ENCODERS,
                                   num_ray_samples=NUM_RAY_SAMPLES, 
                                   lr=LR,
                                   log_every=LOG_EVERY)
else:
    lightning_module = LightningModule(model=main_model,
                                       image_size=IMAGE_SIZE,
                                       is_debug=IS_DEBUG,
                                       num_encoders=NUM_ENCODERS, 
                                       num_viewdir_encoders=NUM_VIEWDIR_ENCODERS,
                                       num_ray_samples=NUM_RAY_SAMPLES, 
                                       lr=LR,
                                       log_every=LOG_EVERY)

logger = TensorBoardLogger("../lightning_logs/", 
                               log_graph=False,
                               version=None)

data_module.setup()
trainer = pl.Trainer(
    # Trainer args
        min_epochs=0,
        max_epochs=NUM_EPOCHS,
        precision=16,
        stochastic_weight_avg=True,
        # gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=1,

    # Other args
        logger=logger,
        log_every_n_steps=50,
        # val_check_interval=0.5,

    # Dev args
        # num_sanity_val_steps=0,
        # fast_dev_run=True, 
        # overfit_batches=4,
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
        # track_grad_norm=2,
        # weights_summary='full',
        # profiler="simple", # "advanced" "pytorch"
        # log_gpu_memory=True,
        gpus=1)

if CHECKPOINT_PATH is None:
    trainer.fit(lightning_module, datamodule=data_module)

trainer.test(lightning_module, datamodule=data_module)
