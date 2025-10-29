# Diffusion Models for X-ray Image Generation

## Overview
We compare and fine-tune three diffusion models (DDPM, DINO, XReal) on the NIH Chest X-ray14 dataset to study generation quality.


### How to Run
```bash
pip install -r requirements.txt
python data/prepare_dataset.py
python train/train_ddpm.py
```

### Team Members
- Adrian Kristanto (123040001)
- Frederick Khasanto (122040014)
- Philip Leong Jun Hwa (125400024)
- Stefan Susanto (122040041)