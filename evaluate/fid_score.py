from torchmetrics.image.fid import FrechetInceptionDistance
import torch

def compute_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()
