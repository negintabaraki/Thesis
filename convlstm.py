import torch.nn as nn
import torch
import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pprint import pprint
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F
from torch.optim import Adam  
import SMP_Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def symmetric_absolute_percentage_error(y_true, y_pred):
    return (2.0 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-7)).mean().item()

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output

class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=1,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # print(X.shape)
        # Forward propagation through all the layers
        output = self.sequential(X)
        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        
        return nn.Sigmoid()(output)

class model(pl.LightningModule):
    def __init__(self):
        super(model, self).__init__()

        # Define the ConvLSTM model
        self.seq2seq = Seq2Seq(
            num_channels=3,
            num_kernels=16,
            kernel_size=3,
            padding=1,
            activation="relu",
            frame_size=(480, 640), # Adjust as per your frame size
            num_layers=3
        )
        # self.loss_fn = nn.BCELoss(reduction='sum')
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        # x['input'] = x['input']/255.0
        # x['base'] = x['base']/255.0
        lstm_out = self.seq2seq(x['input'])
        # print("lstm output shape: ", lstm_out.shape)
        # print(lstm_out.min(), lstm_out.max())

        return lstm_out
    
    def shared_step(self, batch, stage):
        mask = batch["output"]
        # print('mask', mask.shape)
        # print('mask ', mask.min(), mask.max())
        preds = self.forward(batch)
        # print('preds', preds.shape)
        # print('preds ', preds.min(), preds.max())
        loss = self.loss_fn(preds, mask)
        prob_mask = preds.sigmoid()
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
        # Forward pass
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


early_stop_callback = callbacks.EarlyStopping(
    monitor="valid_loss",
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode="min")
