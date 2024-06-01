import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


class MultiVariableDataset(Dataset):
    def __init__(self, df_raw, features, target, seq_len):
        self.data = df_raw
        self.seq_len = seq_len
        self.y = torch.tensor(self.data[target].values).float()
        # self.y = torch.tensor(self.data[target].values).float()
        self.X = torch.tensor(self.data[features].values).float()

    def __len__(self):
        return self.data.shape[0] - self.seq_len -1

    def __getitem__(self, idx):
        features = self.X[idx:(idx + self.seq_len)]
        target = self.y[idx + self.seq_len]  # the return value is numpy float
        target = target.unsqueeze(0)
       
        return features, target

class PassengerDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, seq_len = 1, batch_size = 128, num_workers=0):
        super().__init__()
        self.raw_df = None
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = None #int(len(df_final) * 0.66)
        self.valid_size = None #len(df_final) - train_size
        
        self.features = ['year', 'month', '#Passengers']
        self.target = '#Passengers'
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' and self.raw_df is not None:
            return 
        if stage == 'test' and self.raw_df is not None:
            return
        if stage is None and self.raw_df is not None:  
            return

        dataframe = pd.read_csv('archive/AirPassengers.csv', parse_dates=['Month'])
        dataframe = dataframe.sort_values(by='Month').reset_index(drop=True)
        dataframe['year'] = dataframe['Month'].dt.year
        dataframe['month'] = dataframe['Month'].dt.month
        # nomalization
        df_values = dataframe['#Passengers']
        min_value = df_values.min()
        max_value = df_values.max()
        
        df_norm = (df_values - min_value) / (max_value - min_value)
        df_final = pd.concat([df_norm, dataframe[['year', 'month']]], axis=1)
        self.raw_df = df_final
        self.train_size = int(len(df_final) * 0.66)
        self.valid_size = len(df_final) - self.train_size
        print('setup is finished')
        
    def train_dataloader(self):
        train_dataset = MultiVariableDataset(self.raw_df.loc[0:self.train_size,:], self.features, self.target, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = MultiVariableDataset(self.raw_df.loc[self.train_size:,:], self.features, self.target, self.seq_len)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return val_loader
    # for now the test dataloader is the same as the valid dataloader
    def test_dataloader(self):
        test_dataset = MultiVariableDataset(self.raw_df.loc[self.train_size:,:], self.features, self.target, self.seq_len)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return test_loader

class LSTMRegressor(pl.LightningModule):

    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout
                 ):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        return y_pred

class TrainingSteps(pl.LightningModule):

    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.learning_rate = 0.001
        self.criterion = nn.MSELoss()
    # --------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        log = {'val_loss': loss}
        return {'val_loss': loss, 'log': log}
      
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.criterion(y_hat, y)
        # result = pl.EvalResult()
        # result.log('test_loss', loss)
        return {'loss':test_loss}

if __name__ == "__main__":
    log_path = Path('.')
    MODEL = log_path/'model_v1'
    MODEL.mkdir(exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=MODEL,
        filename='m-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    trainer = Trainer(
        # gpus=0,
        max_epochs=100,
        logger=False,
        callbacks=[checkpoint_callback],
    )

    dm = PassengerDataModule(4, 4, 0)
    model = LSTMRegressor(3, 64, 4, 4, 2, 0.2)
    training_steps = TrainingSteps(model)
    trainer.fit(training_steps, dm)