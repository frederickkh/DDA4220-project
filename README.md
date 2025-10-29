# Diffusion Models for X-ray Image Generation

## Overview
We compare and fine-tune three diffusion models (DDPM, DINO, XReal) on the NIH Chest X-ray14 dataset to study generation quality.

### Team Roles
- Fred — DDPM + Evaluation
- Philip — DINO Diffusion
- Adrian — XReal Diffusion
- Stefan — Dataset and Preprocessing

### How to Run
```bash
pip install -r requirements.txt
python data/prepare_dataset.py
python train/train_ddpm.py

