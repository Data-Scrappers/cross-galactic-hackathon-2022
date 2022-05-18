from torch import nn

from decoder import Decoder
from encoder import Encoder


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, device, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
