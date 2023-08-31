
import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pprint import pprint
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.loggers import CSVLogger
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

def symmetric_absolute_percentage_error(y_true, y_pred):
    return (2.0 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-7)).mean().item()

def sad_diff(predict, alpha):
    return (torch.sum(torch.abs(predict - alpha)) / 1000).mean()

def mse_diff(predict, alpha):
    pixel = predict.shape[0] * predict.shape[1]
    return (torch.sum((predict - alpha) ** 2) / pixel).mean()

def mad_diff(predict, alpha):
    pixel = predict.shape[0] * predict.shape[1]
    return (torch.sum(torch.abs(predict - alpha)) / pixel).mean()


class Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        # self.model = smp.Unet(
        #     encoder_name="resnet34",
        #     encoder_weights="imagenet",
        #     in_channels=7,         # Updated to 7 as per your requirements
        #     classes=1              # Updated to 1 as per your requirements
        # )
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_classes, **kwargs
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.encoder_name = encoder_name
        
    def forward(self, image):

        image = image/255.0
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["input"]

        if image.ndim != 4:
            raise ValueError(f"Image should have 4 dimensions, but got {image.ndim}")

        h, w = image.shape[2:]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError("Image height and width should be divisible by 32")

        mask = batch["output"]
        if mask.ndim != 4 or not (mask.max() <= 1.0 and mask.min() >= 0):
            raise ValueError("Invalid mask")

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Calculate mse, mad
        mse_loss = mean_squared_error(pred_mask, mask)
        mad_loss = mean_absolute_error(pred_mask, mask)
        sad_loss = symmetric_absolute_percentage_error(pred_mask, mask)

        self.log(f'{stage}_loss', loss)
        self.log(f'{stage}_mse', mse_loss)
        self.log(f'{stage}_mad', mad_loss)
        self.log(f'{stage}_sad', sad_loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")

        image = batch["input"]
        mask = batch["output"]

        # Base prediction is the last channel in the input image
        base_pred = image[:, -1, :, :].unsqueeze(1)

        # Calculate the losses
        base_dice_loss = self.loss_fn(base_pred, mask)
        base_mse_loss = mean_squared_error(base_pred, mask)
        base_mad_loss = mean_absolute_error(base_pred, mask)
        base_sad_loss = symmetric_absolute_percentage_error(base_pred, mask)

        # Log the losses
        self.log('test_base_dice_loss', base_dice_loss)
        self.log('test_base_mse_loss', base_mse_loss)
        self.log('test_base_mad_loss', base_mad_loss)
        self.log('test_base_sad_loss', base_sad_loss)

        return loss

    def configure_optimizers(self):
        # Use AdamW instead of Adam, it's generally better
        return torch.optim.AdamW(self.parameters(), lr=0.0001)


# Early stopping based on validation loss

early_stop_callback = callbacks.EarlyStopping(
    monitor="valid_loss",
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode="min") 