import torch
from torch import nn
import torch.nn.functional as F
from config import *

class VAEFlat(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), # (B, 32, 128, 128)
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 128, 4, 2, 1, bias=False), # (B, 128, 64, 64)
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), # (B, 128, 32, 32)
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), # (B, 128, 16, 16)
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), # (B, 128, 8, 8)
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), # (B, 128, 4, 4)
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), # (B, 128, 2, 2)
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        self.conv_mu = nn.Linear(128 * 2 * 2, 256)
        self.conv_logvar = nn.Linear(128 * 2 * 2, 256)
        self.dec_unflatten = nn.Linear(256, 128 * 2 * 2)
        self.dec_bn = nn.BatchNorm1d(128 * 2 * 2)
        self.dec_act = nn.LeakyReLU()
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder_conv(x) # (B, C, D, H)
        mu, logvar = self.conv_mu(h), self.conv_logvar(h)
        z = self.reparameterize(mu, logvar) # Latent (B, C, D, H)
        z = self.dec_act(self.dec_bn(self.dec_unflatten(z)))
        z = z.view(-1, 128, 2, 2)
        recon = self.decoder_conv(z)
        return recon, mu, logvar