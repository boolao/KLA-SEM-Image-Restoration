# KLA Hackathon: SEM Image Restoration

This repository contains the complete training pipeline and evaluation script for restoring degraded SEM images using a custom SR-UNet Residual architecture.

## Repository Structure
* `run.py` — **The Evaluation Script**. Automated inference engine.
* `training_pipeline.ipynb` — **The Training Script**. Contains the 80-epoch base training and 10-epoch fine-tuning.
* `models/sr_unet_true_ultimate.pth` — **Trained Model Weights**.
* `Restored_Test_Outputs/` — **Restored Test Outputs**.
* `requirements.txt` — Environment dependencies.

## Setup Instructions

**Prerequisites:** Ensure you have Python and Git installed on your system.

**1. Clone the repository**
Open your terminal (or Git Bash) and run the following command to download the project:
```bash
git clone [https://github.com/boolao/KLA-SEM-Image-Restoration.git](https://github.com/boolao/KLA-SEM-Image-Restoration.git)

cd KLA-SEM-Image-Restoration

pip install -r requirements.txt
