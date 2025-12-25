import torch
from vae import VAE
from vae_new import VAEFlat
from config import *
import os
from PIL import Image
from torchvision.utils import save_image
from dataloader import ImageDataset
from tqdm import tqdm
from torch.utils.data import DataLoader

def sample_latents():
    with torch.no_grad():
        z = torch.randn(10, 128, 2, 2).to(device)
        # z = vae.dec_act(vae.dec_bn(vae.dec_unflatten(z))).view(-1, 128, 2, 2)
        samples = vae.decoder_conv(z)
        save_image(
            samples,
            f"{latent_dir}/random_samples.png",
            nrow=4
        )

def reconstruct():
    with torch.no_grad():
        image_paths = [os.path.join(resized_img_dir, path) for path in sorted(os.listdir(resized_img_dir))[:8] if path.endswith(".jpg")]
        recon_loader = DataLoader(ImageDataset(image_paths), batch_size=vae_batch_size, shuffle=True)
        for x in tqdm(recon_loader, desc=f"Progress", colour="#00FFEF"):
            recon_x, _, _ = vae(x)
            save_image(torch.cat([x, recon_x]), f"{latent_dir}/recon_2.png",nrow=x.size(0))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(latent_dir, exist_ok=True)
    # vae = VAE().to(device)
    vae = VAEFlat().to(device)
    # path = f"{vae_checkpoint_dir}/{vae_weight}.pth"
    path = f"{vae_checkpoint_dir}/test_best_128_3_32_128_128_128_128_128_128_beta_1_tvl_0.001_batch_256.pth"
    path = f"{vae_checkpoint_dir}/test_best_128_3_32_128_128_128_128_128_128_beta_1_tvl_0.001_batch_32_lr_0.0001.pth"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()
    reconstruct()
    # sample_latents()