import torch
from torch import nn


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, device):
        super().__init__()
        self.num_sensors = num_sensors  # or actuators
        self.hidden_units = hidden_units
        self.num_layers = 1
        self.device = device

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.lstm = self.lstm.to(self.device)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        self.linear = self.linear.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()

        return out
