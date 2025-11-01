# # !pip install diffusers
# from diffusers import DDPMPipeline

# model_id = "google/ddpm-cifar10-32"

# # load model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)

# # run pipeline in inference (sample random noise and denoise)
# image = ddpm().images[0]

# # save image
# image.save("ddpm_generated_image.png")


from diffusers import UNet2DModel, DDPMScheduler
import torch
import torch.nn.functional as F

class DDPM(torch.nn.Module):
    def __init__(self, image_size=128, timesteps=1000):
        super().__init__()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, timesteps, noise):
        noisy = self.noise_scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.model(noisy, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples=16, device="cuda"):
        self.model.eval()
        image_size = self.model.config.sample_size
        
        samples = torch.randn((num_samples, 3, image_size, image_size), device=device)
        
        num_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_timesteps, device=device)

        for t in self.noise_scheduler.timesteps:
            
            # The UNet expects a tensor for t, shaped (batch_size,)
            t_tensor = t.repeat(num_samples)
            
            # Predict the noise residual (epsilon)
            residual = self.model(samples, t_tensor).sample
            
            # Denoise the image one step using the scheduler
            # NOTE: scheduler.step() expects t to be a scalar/tensor, not the repeated tensor
            samples = self.noise_scheduler.step(residual, t, samples).prev_sample
            
        # Clamp and normalize to [0,1] range for saving
        return (samples.clamp(-1, 1) + 1) / 2
