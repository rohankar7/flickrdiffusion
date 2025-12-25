import torch
from torch import nn
import torch.nn.functional as F
from config import *

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # 1×1×1 convs give channel‑wise μ and log σ², shape = (B, latent_channels, 4, 4, 4)
        self.conv_mu = nn.Conv2d(vae_latent_channels, vae_latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(vae_latent_channels, vae_latent_channels, kernel_size=1)
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
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
        recon = self.decoder_conv(z)
        return recon, mu, logvar