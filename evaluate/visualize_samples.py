import matplotlib.pyplot as plt
import torch

def visualize_samples(images, title="Generated Samples"):
    grid = torch.cat(list(images[:8]), dim=2).squeeze().cpu()
    plt.imshow(grid, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()
