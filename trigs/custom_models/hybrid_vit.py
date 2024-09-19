import numpy as np
from torchvision import models
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class HybridTranformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the classifier
        :param num_classes: int, the number of classes
        :param latent_dim: int, the size of the latent dimension, which is expected to be the size of the input
        embeddings and will be also the size of the output embedding
        :param num_heads: int, number of heads in each transformer layer
        :param num_layers: int, number of transformer layers in the encoder
        :param dropout: float, dropout rate, used to set up the transformer encoder and to the positional encoding
        """
        super(HybridTranformer, self).__init__()
        self.latent_dim = latent_dim
        self.resnet = models.resnext50_32x4d(num_classes=latent_dim, pretrained=False)
        self.z_query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dropout=dropout
        )
        self.trans_encoder = nn.TransformerEncoder(
            trans_encoder_layer, num_layers=num_layers
        )
        self.logits = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Model evaluation entry point
        :param x: seq_length x batch_size x latent_dim tensor
        :return: The classification logits
        """
        embedding = torch.empty(size=(x.shape[0], x.shape[1], self.latent_dim)).to(x.device)
        for idx, signatures in enumerate(x):
            embedding[idx] = self.resnet(signatures)
        embedding = embedding.transpose(dim0=1, dim1=0)
        xseq = torch.cat((self.z_query.expand(-1, x.shape[0], -1), embedding), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        y = self.trans_encoder(xseq)
        z = y[0]
        logits = self.logits(z)
        return logits


def hybrid_transfomer(num_classes=2, latent_dim=256, num_heads=4, num_layers=3):
    return HybridTranformer(num_classes=num_classes, latent_dim=latent_dim, num_heads=num_heads, num_layers=num_layers)
