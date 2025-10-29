# main.py

import argparse
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model.ddpm import DDPM


def get_dataloader(batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, epochs, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, model.noise_scheduler.num_train_timesteps, (x.shape[0],), device=device)

            # Forward pass
            noise_pred = model(x, timesteps, noise)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Step {i}: Loss = {loss.item():.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"checkpoints/ddpm_epoch_{epoch+1}.pt")


def sample(model, device, num_samples=16, save_path="samples.png"):
    samples = model.sample(num_samples=num_samples, device=device)
    save_image(samples, save_path, nrow=4)
    print(f"✅ Samples saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="DDPM Trainer/Inference using diffusers")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = DDPM(image_size=args.image_size, timesteps=args.timesteps).to(device)

    if args.checkpoint:
        print(f"🔄 Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.mode == "train":
        dataloader = get_dataloader(args.batch_size, args.image_size)
        train(model, dataloader, args.epochs, args.lr, device)
    elif args.mode == "sample":
        sample(model, device, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
