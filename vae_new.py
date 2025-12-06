import torch
from torch import nn
import torch.nn.functional as F
from config import *

class VAELinear(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.ReLU(), # 128
            nn.Conv2d(32, 128, 4, 2, 1, bias=False), nn.ReLU(), # 64
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.ReLU(), # 32
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.ReLU(),  # 16
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.ReLU(),  # 8
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.ReLU(),  # 4
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.ReLU(),  # 2
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.ReLU(),  # 1
        )
        self.fc_mu = nn.Linear(128 * 1 * 1, vae_latent_channels)
        self.fc_logvar = nn.Linear(128 * 1 * 1, vae_latent_channels)
        self.decoder_in = nn.Linear(vae_latent_channels, 128 * 1 * 1)
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),  # 2
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),  # 4
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(), # 64
            nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.ReLU(), # 128
            nn.ConvTranspose2d(32, 3, 3, 1, 1), nn.ReLU(), # 128
            nn.Sigmoid()
        )
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        print("HAHA", h.shape)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h = self.decoder_in(z)
        h = h.view(h.size(0), 128, 1, 1)
        h = self.decoder_conv(h) # reconstruction
        return h, mu, logvar
def get_annealed_beta(epoch, warmup_epochs=100, max_beta=vae_beta_kld):
    return max_beta * min(1, epoch / warmup_epochs)
def vae_loss(recon_x, x, mu, logvar, beta_kld):
    bce = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + (beta_kld * kld), bce, kld, 0