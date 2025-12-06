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
            nn.GroupNorm(vae_group_size, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(vae_group_size, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(vae_group_size, 128), nn.ReLU(inplace=True),
        )
        # 1×1×1 convs give channel‑wise μ and log σ², shape = (B, latent_channels, 4, 4, 4)
        self.conv_mu = nn.Conv2d(vae_latent_channels, vae_latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(vae_latent_channels, vae_latent_channels, kernel_size=1)
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.GroupNorm(vae_group_size, 128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.GroupNorm(vae_group_size, 32), nn.ReLU(inplace=True),
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
def total_variance_loss(x):
    tvl_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).mean() # TV-L2
    tvl_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).mean() # TV-L2
    return (tvl_h + tvl_w)
def get_annealed_beta(epoch, warmup_epochs=100, max_beta=vae_beta_kld):
    return max_beta * min(1, epoch / warmup_epochs)
def vae_loss(recon_x, x, mu, logvar, beta_kld):
    # bce = F.binary_cross_entropy(recon_x, x, reduction="mean")
    bce = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    tvl = total_variance_loss(recon_x)
    return bce + kld*beta_kld + tvl*vae_lambda_tvl, bce, kld, tvl