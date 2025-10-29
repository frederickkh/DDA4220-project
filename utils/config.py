CONFIG = {
    "image_size": 128,
    "batch_size": 16,
    "epochs": 20,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
