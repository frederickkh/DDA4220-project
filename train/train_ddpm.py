import torch
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.ddpm import DDPM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare dataset (e.g., CIFAR10)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = datasets.CIFAR10(root="./data", download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
ddpm = DDPM(image_size=64, channels=3).to(device)
optimizer = Adam(ddpm.parameters(), lr=1e-4)

# Training loop
for epoch in range(1):
    for x, _ in loader:
        x = x.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, ddpm.noise_scheduler.num_train_timesteps, (x.shape[0],), device=device)
        noise_pred = ddpm(x, timesteps, noise)

        loss = nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
