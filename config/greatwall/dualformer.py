from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.greatwall_dataset import *
from geoseg.losses.dice_bce_loss import dice_bce_loss
from geoseg.models.DUALFormer import ft_demformer

from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 80
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES
    
weights_name = "dualformer-512-greatwall-e80"
weights_path = "model_weights/greatwall/{}".format(weights_name)
test_weights_name = "last"
log_name = "greatwall/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # the path for the pretrained model weight
gpus = "auto"  # default or gpu ids:[0] or gpu nums: 2
resume_ckpt_path = None  # whether continue training with the checkpoint, default None


use_aux_loss = False
use_connectivity_loss = True
connectivity_loss = con_loss()
connectivity_loss_weight = 0.3
connectivity_loss_ratio = (0.5, 0.5)

use_dem = True
use_ridge = True
enable_aug_images = True
aug_images_dir = "images_aug"
aug_images_list = "kept_images.txt"


# define the network
net = ft_demformer(num_classes=num_classes, decoder_channels=256, dem_in_channels=1, use_connectivity_head=use_connectivity_loss, connectivity_channels=9)

# define the loss
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)



# define the dataloader
train_dataset = GreatWallDataset(
    split="train",
    transform=train_aug,
    mosaic_ratio=0.0,
    use_dem=use_dem,
    use_ridge=use_ridge,
    use_connectivity=use_connectivity_loss,
    enable_aug_images=enable_aug_images,
    aug_images_dir=aug_images_dir,
    aug_images_list=aug_images_list,
)
val_dataset = GreatWallDataset(
    split="val",
    transform=val_aug,
    mosaic_ratio=0.0,
    use_dem=use_dem,
    use_ridge=use_ridge,
    use_connectivity=use_connectivity_loss,
)
test_dataset = val_dataset

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
