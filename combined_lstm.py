import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import mean_squared_error, mean_absolute_error
import SMP_Model # Your module
import segmentation_models_pytorch as smp
import convlstm
def symmetric_absolute_percentage_error(y_true, y_pred):
    return (2.0 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-7)).mean().item()


class Combined_lstm(pl.LightningModule):
    def __init__(self):
        super(Combined_lstm, self).__init__()
        self.model_lstm = convlstm.model()
        self.model_lstm.load_state_dict(torch.load('lstm_checkpoints/checkpoint-lstm-5e.ckpt')['state_dict'])
       
        for param in self.model_lstm.parameters():
            param.requires_grad = False

            
        self.combine_conv = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 1, kernel_size=1),
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, batch):
        output_lstm = self.model_lstm(batch)
        # print(output_lstm.shape)
        # Weighted sum of the outputs
        base_pred = batch["base"]
        out = torch.cat([output_lstm, base_pred], dim=1)
        # print(out.shape)
        out = self.combine_conv(out)
        return out

    def shared_step(self, batch, stage):

        mask = batch["output"]
        
        logits_mask = self.forward(batch)
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
        base_pred = batch["base"]
        mask = batch["output"]

        base_dice_loss = self.loss_fn(base_pred, mask)
        base_mse_loss = mean_squared_error(base_pred, mask)
        base_mad_loss = mean_absolute_error(base_pred, mask)
        base_sad_loss = symmetric_absolute_percentage_error(base_pred, mask)
        
        self.log('test_base_dice_loss', base_dice_loss)
        self.log('test_base_mse_loss', base_mse_loss)
        self.log('test_base_mad_loss', base_mad_loss)
        self.log('test_base_sad_loss', base_sad_loss)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)