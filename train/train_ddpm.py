# import torch
# from torch.optim import Adam
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from models.ddpm import DDPM

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Prepare dataset (e.g., CIFAR10)
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])
# dataset = datasets.CIFAR10(root="./data", download=True, transform=transform)
# loader = DataLoader(dataset, batch_size=64, shuffle=True)

# # Initialize model
# ddpm = DDPM(image_size=64, channels=3).to(device)
# optimizer = Adam(ddpm.parameters(), lr=1e-4)

# # Training loop
# for epoch in range(1):
#     for x, _ in loader:
#         x = x.to(device)
#         noise = torch.randn_like(x)
#         timesteps = torch.randint(0, ddpm.noise_scheduler.num_train_timesteps, (x.shape[0],), device=device)
#         noise_pred = ddpm(x, timesteps, noise)

#         loss = nn.functional.mse_loss(noise_pred, noise)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f"Loss: {loss.item():.4f}")

# train_ddpm.py
import torch
from torch.optim import Adam
from tqdm import tqdm
from models.ddpm import DDPM
from utils.dataset_loader import get_dataloader
from torchvision.utils import save_image
import os

def train_ddpm(data_dir, epochs=5, batch_size=8, lr=1e-4, image_size=128, device="cuda"):
    dataloader = get_dataloader(data_dir, batch_size, image_size)
    print("Dataset loaded")
    model = DDPM(image_size=image_size).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            print("Entering training step")
            imgs = imgs.to(device)
            noise = torch.randn_like(imgs)
            timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (imgs.shape[0],), device=device)

            loss = model(imgs, timesteps, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print("Finish one image")

        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/ddpm_epoch_{epoch+1}.pt")

        # Generate samples
        samples = model.sample(num_samples=8, device=device)
        save_image(samples, f"samples/sample_epoch_{epoch+1}.png", nrow=4)

    print("✅ Training complete!")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data/images"  # path to extracted NIH dataset
    train_ddpm(data_dir=data_dir, epochs=2, batch_size=8, device=device)
