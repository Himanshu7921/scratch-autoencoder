from .layers import LinearLayer
import torch
import torch.nn as nn
from dataclasses import dataclass

class Encoder(nn.Module):
    def __init__(self, input_message_dim: int, hidden_dim: int, encoded_message_dim: int):
        super().__init__()
        self.linear_1 = LinearLayer(input_message_dim, hidden_dim)
        self.linear_2 = LinearLayer(hidden_dim, encoded_message_dim)
    
    def forward(self, message: torch.tensor):
        encoded_message = self.linear_2(torch.sigmoid(self.linear_1(message)))
        return encoded_message

class Decoder(nn.Module):
    def __init__(self, encoded_msg_dim: int, hidden_dim: int, decoded_msg_dim: int):
        super().__init__()
        self.linear_1 = LinearLayer(encoded_msg_dim, hidden_dim)
        self.linear_2 = LinearLayer(hidden_dim, decoded_msg_dim)

    def forward(self, encoded_msg: torch.tensor):
        decoded_message = torch.sigmoid(self.linear_2(torch.sigmoid(self.linear_1(encoded_msg))))
        return decoded_message
    
class AutoEncoder(nn.Module):
    def __init__(self, input_message_dim: int,
                        enc_hidden_dim: int,
                        encoded_message_dim: int,
                        dec_hidden_dim: int):
        super().__init__()
        self.encoder = Encoder(input_message_dim = input_message_dim, hidden_dim = enc_hidden_dim, encoded_message_dim = encoded_message_dim)
        self.decoder = Decoder(encoded_msg_dim = encoded_message_dim, hidden_dim = dec_hidden_dim, decoded_msg_dim = input_message_dim)

    def forward(self, x: torch.tensor):
        self.latent_representation = self.encoder(x)
        return self.decoder(self.latent_representation)