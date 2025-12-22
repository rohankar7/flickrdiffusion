import torch
from vae import *
from config import *
import os
from PIL import Image
from torchvision.utils import save_image
from dataloader import ImageDataset
from tqdm import tqdm
from torch.utils.data import DataLoader

def sample_latents():
    os.makedirs(latent_dir, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(10, 256, 16, 16).to(device)
        samples = vae.decoder_conv(z)
        save_image(
            samples,
            f"{latent_dir}/random_samples.png",
            nrow=4
        )

def reconstruct():
    with torch.no_grad():
        image_paths = [os.path.join(resized_img_dir, path) for path in os.listdir(resized_img_dir)[:16] if path.endswith(".jpg")]
        recon_loader = DataLoader(ImageDataset(image_paths), batch_size=vae_batch_size, shuffle=True)
        for x in tqdm(recon_loader, desc=f"Progress", colour="#00FFEF"):
            recon_x, _, _ = vae(x)
            save_image(torch.cat([x, recon_x]), f"{latent_dir}/recon.png",nrow=x.size(0))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    checkpoint = torch.load(f"{vae_checkpoint_dir}/{vae_weight}.pth", map_location=device, weights_only=True)
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()
    reconstruct()
    # sample_latents()