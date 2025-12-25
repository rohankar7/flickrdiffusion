import torch
from torch import nn, optim
import torch.nn.functional as F
from dataloader import image_dataloader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
from config import *
from vae import VAE
import matplotlib.pyplot as plt
from seed import seed_everything

def total_variance_loss(x):
    tvl_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).mean() # TV-L2
    tvl_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).mean() # TV-L2
    return (tvl_h + tvl_w)

def get_annealed_beta(epoch, warmup_epochs=100, max_beta=vae_beta_kld): return max_beta * min(1, epoch / warmup_epochs)

def vae_loss(recon_x, x, mu, logvar, beta_kld):
    bce = F.l1_loss(recon_x, x, reduction="mean")
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    tvl = total_variance_loss(recon_x)
    return bce + kld*beta_kld + tvl*vae_lambda_tvl, bce, kld, tvl

def train_test_vae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = image_dataloader()
    # optimizer = optim.Adam(vae.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=1e-2)
    vae = VAE().to(device)
    # vae = VAEFlat().to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=vae_optim_lr, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, threshold=0.001)
    os.makedirs(f"{vae_checkpoint_dir}", exist_ok=True)
    num_epochs = vae_num_epochs
    early_stopping_patience = vae_stopping_patience
    early_stopping_counter = 0
    best_test_loss = float("inf")
    train_bce_loss = []
    train_kld_loss = []
    # Train VAE
    # analyze_voxel_sparsity(train_loader)
    for epoch in range(num_epochs):
        vae.train()
        train_loss, total_bce, total_kld, total_tvl = 0, 0, 0, 0
        for image_gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", colour="#00FFEF"):
            x = image_gt.to(device)
            recon_x, mu, logvar = vae(x)
            loss, bce, kld, tvl = vae_loss(recon_x, x, mu, logvar, beta_kld=get_annealed_beta(epoch))
            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_kld = total_kld / len(train_loader)
        train_bce_loss.append(avg_bce)
        train_kld_loss.append(avg_kld)
        # print(f"🔥 Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f} | Recon: {avg_bce:.6f} | KLD: {avg_kld:.6f} | TVL: {avg_tvl:.6f}")
        print(f"🔥 Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f} | Recon: {avg_bce:.6f} | KLD: {avg_kld:.6f}")
        # Test VAE
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for image_gt in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}", colour="#FFDD22"):
                x = image_gt.to(device)
                recon_x, mu, logvar = vae(x)
                loss, _, _, _ = vae_loss(recon_x, x, mu, logvar, beta_kld=vae_beta_kld)
                test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            scheduler.step(avg_test_loss)
            print(f"🧪 Test Loss = {avg_test_loss:.6f}")
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({
                    "epoch": epoch,
                    "vae": vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "test_loss": best_test_loss
                }, f"{vae_checkpoint_dir}/{vae_weight}.pth")
                print(f"Saved best model at {epoch+1}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
        torch.cuda.empty_cache()
    # plot_recon_vs_kld(train_bce_loss, train_kld_loss, train_tvl_loss)
    plot_recon_vs_kld(train_bce_loss[2:], train_kld_loss[2:])

# def plot_recon_vs_kld(train_bce_loss, train_kld_loss, train_tvl_loss):
def plot_recon_vs_kld(train_bce_loss, train_kld_loss):
    epochs = list(range(len(train_bce_loss)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_bce_loss, label='Reconstruction Loss (BCE)', color='blue', linewidth=2)
    plt.plot(epochs, train_kld_loss, label='KL Divergence', color='red', linewidth=2)
    # plt.plot(epochs, train_tvl_loss, label='TVL', color='yellow', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss vs KL Divergence over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    seed_everything(42)
    train_test_vae()