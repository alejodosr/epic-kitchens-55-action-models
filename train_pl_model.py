import argparse
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize
from task_utils.utils import *
from task_utils.datasets import KFCDataset
import pandas as pd
import time
from archs.tsn_pl import TSN
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


parser = argparse.ArgumentParser(
    description="Action recognition training")
parser.add_argument('--train_csv', type=str, default="./videos/satis_task_b_dataset/train/train.csv")
parser.add_argument('--val_csv', type=str, default="./videos/satis_task_b_dataset/val/val.csv")
parser.add_argument('--batch', default=1, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--segment_count', default=8, type=int)
parser.add_argument('--repo', type=str, default="epic-kitchens/action-models")
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--num_epochs', default=60, type=int)
parser.add_argument('--validation_epochs', default=2, type=int)

args = vars(parser.parse_args())

# Check if CUDA is available
if args['gpu_id'] >= 0 and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Model instance
model = TSN(args)

# Train transform
cropping = Compose([
    GroupScale(model.tsn_model.scale_size),
    GroupCenterCrop(model.tsn_model.input_size),
])
train_transform = Compose([
    cropping,
    Stack(roll=args['base_model'] == args['base_model']),
    ToTorchFormatTensor(div=args['base_model'] != args['base_model']),
    GroupNormalize(model.tsn_model.input_mean, model.tsn_model.input_std),
])

# Val transform
cropping = GroupOverSample(model.tsn_model.input_size, model.tsn_model.scale_size)
val_transform = Compose([
    cropping,
    Stack(roll=args['base_model'] == args['base_model']),
    ToTorchFormatTensor(div=args['base_model'] != args['base_model']),
    GroupNormalize(model.tsn_model.input_mean, model.tsn_model.input_std),
])

# Datasets
train_dataset = KFCDataset(args['train_csv'], segment_count=args['segment_count'], transform=train_transform, debug=True)
val_dataset = KFCDataset(args['val_csv'], segment_count=args['segment_count'], transform=val_transform, debug=True)

# Set datasets
model.set_train_dataset(train_dataset)
model.set_val_dataset(val_dataset)

checkpoint_callback = ModelCheckpoint(
    save_top_k=True,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(gpus=[args['gpu_id']],
                     check_val_every_n_epoch=args['validation_epochs'],
                     max_epochs=args['num_epochs'],
                     checkpoint_callback=checkpoint_callback
                     )
trainer.fit(model)