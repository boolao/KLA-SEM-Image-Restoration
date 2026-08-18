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

## 🛠️ Setup & Installation

**Prerequisites:** Ensure you have Python and Git installed on your system.

**1. Clone the repository**
```bash
git clone https://github.com/boolao/KLA-SEM-Image-Restoration.git
```

**2. Navigate into the project directory**
```bash
cd KLA-SEM-Image-Restoration
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
## 🚀 How to Run Inference

To evaluate the model on a new set of degraded `.npy` images, use the `run.py` script. The script requires two arguments: the folder containing the noisy inputs, and the folder where you want the restored images saved.

```bash
python run.py <path_to_input_directory> <path_to_output_directory>
```

**Example Run:**
```bash
python run.py ./hidden_test_data/NoisyLR ./submission_outputs
```

### Key Technical Features of `run.py`:
* **Automatic Device Detection:** Automatically utilizes CUDA (GPU) if available, with a seamless fallback to CPU.
* **Dimension Safety Net:** Features dynamic reflection padding to prevent U-Net crashes on odd-sized or non-standard image dimensions.
* **Math Protection:** Built-in `nan_to_num` filters and pixel value clamping ensure zero runtime crashes from extreme noise spikes.
