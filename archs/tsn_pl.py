import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class TSN(pl.LightningModule):

    def __init__(self, hparams):
        super(TSN, self).__init__()

        # Store hparams as a member
        self.hparams = hparams

        # Load model from pytorch hub
        self.tsn_model = torch.hub.load(hparams['repo'], 'TSN', (125, 352), hparams['segment_count'], 'RGB',
                             base_model=hparams['base_model'], pretrained='epic-kitchens', force_reload=True)

        # Training
        self.batch = hparams['batch']
        self.num_workers = hparams['num_workers']
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_loss = 0


    def forward(self, x):
        verb_logits, noun_logits = self.tsn_model(x)
        return verb_logits, noun_logits

    def training_step(self, batch, batch_idx):
        x, verb_targets, noun_targets = batch
        verb_logits, noun_logits = self.forward(x)
        loss = (self.criterion(verb_logits, verb_targets) + self.criterion(noun_logits, noun_targets))
        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, verb_targets_, noun_targets_ = batch
        verb_targets = (torch.ones(10, dtype=torch.long).to(self.device) * verb_targets_)
        noun_targets = (torch.ones(10, dtype=torch.long).to(self.device) * noun_targets_)
        with torch.no_grad():
            verb_logits, noun_logits = self.forward(x)
            loss = (self.criterion(verb_logits, verb_targets) + self.criterion(noun_logits, noun_targets))
            # self.val_loss += loss

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # tensorboard_logs = {'val_loss': loss}

        # return {'val_loss': loss, 'log': tensorboard_logs}
        return {}

    # def validation_end(self, outputs):
    #     val_loss = self.val_loss / len(self.val_dataset)
    #     self.val_loss = 0
    #     tensorboard_logs = {'val_loss': val_loss, 'val_loss_avg': val_loss}
    #
    #     return {'val_loss': val_loss, 'val_loss_avg': val_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def set_val_dataset(self, val_dataset):
        self.val_dataset = val_dataset

    def set_train_dataset(self, train_dataset):
        self.train_dataset = train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          self.batch, num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          1, num_workers=1, shuffle=False)

    def reset_model(self):
        # Load model from pytorch hub
        self.tsn_model = torch.hub.load(self.hparams['repo'], 'TSN', (125, 352), self.hparams['segment_count'], 'RGB',
                             base_model=self.hparams['base_model'], pretrained='epic-kitchens', force_reload=True)

