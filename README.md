# KLA Hackathon: SEM Image Restoration

This repository contains the complete training pipeline and evaluation script for restoring degraded SEM images using a custom SR-UNet Residual architecture.

## Repository Structure
* `run.py` — **The Evaluation Script** (Fulfills Requirement #2). Automated inference engine.
* `training_pipeline.ipynb` — **The Training Script** (Fulfills Requirement #3). Contains the 80-epoch base training and 10-epoch fine-tuning.
* `models/sr_unet_true_ultimate.pth` — **Trained Model Weights** (Fulfills Requirement #4).
* `Restored_Test_Outputs/` — **Restored Test Outputs** (Fulfills Requirement #5).
* `requirements.txt` — Environment dependencies.

## Setup Instructions

1. Clone this repository:
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
