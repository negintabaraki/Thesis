import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import mean_squared_error, mean_absolute_error
import SMP_Model # Your module
import segmentation_models_pytorch as smp

def symmetric_absolute_percentage_error(y_true, y_pred):
    return (2.0 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-7)).mean().item()    
    
class BoostingModel(pl.LightningModule):
    def __init__(self):
        super(BoostingModel, self).__init__()
        
        self.model_DEEPLABV3 = SMP_Model.Model("DeepLabV3", "resnet50", "imagenet", in_channels=7, out_classes=1)
        self.model_PAN = SMP_Model.Model("PAN", "resnet50", "imagenet", in_channels=7, out_classes=1)
        self.model_PAN.load_state_dict(torch.load('fully_finetuned_checkpoints/checkpoint-PAN-5e.ckpt')['state_dict'])
        
        for param in self.model_PAN.parameters():
            param.requires_grad = False
\
        
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
    def forward(self, x):
        x1 = self.model_PAN(x)
        x2 = self.model_DEEPLABV3(torch.cat([x, x1], dim=1))
        # print('x3 shape:', x3.shape)
        return x2

    def shared_step(self, batch, stage):
        image = batch["input"]
        mask = batch["output"]
        
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
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)