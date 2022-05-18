from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import torch
import torch.nn as nn

from recurrent_autoencoder import RecurrentAutoencoder


class ModelWrapper:
    def __init__(self, seq_len, n_features, device):
        self.device = device
        self.model = RecurrentAutoencoder(seq_len, n_features, device, 128)
        self.model = self.model.to(self.device)
        # Initialize learning parameters
        self.criterion = nn.L1Loss(reduction='sum').to(device)
        self.num_epochs = 50
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_dataset, validation_dataset, save_model=True):
        history = dict(train=[], val=[])
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 10000.0
        for epoch in range(1, self.num_epochs + 1):
            self.model = self.model.train()
            train_losses = []
            for seq_true in train_dataset:
                self.optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                loss = self.criterion(seq_pred, seq_true)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            val_losses = []
            self.model = self.model.eval()
            with torch.no_grad():
                for seq_true in validation_dataset:
                    seq_true = seq_true.to(self.device)
                    seq_pred = self.model(seq_true)
                    loss = self.criterion(seq_pred, seq_true)
                    val_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train'].append(train_loss)
            history['val'].append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        self.model.load_state_dict(best_model_wts)

        if save_model:
            torch.save(self.model.state_dict(), './model_from_last_train.pth')

        return self.model.eval(), history
