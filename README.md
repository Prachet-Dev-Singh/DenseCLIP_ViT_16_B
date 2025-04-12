# DenseCLIP Segmentation (Standalone Implementation)

## Overview
This project provides a standalone implementation of the DenseCLIP model for semantic segmentation, primarily targeting the Cityscapes dataset. It is designed to run without dependencies on the `mmcv` and `mmsegmentation` libraries.

## Features
- Standalone DenseCLIP implementation
- Training and validation scripts
- Pre-trained CLIP weights support (RN50 backbone)
- Cityscapes dataset support
- Albumentations for data augmentation
- TorchMetrics for evaluation (mIoU, Pixel Accuracy)
- YAML configuration files
- Checkpoint resuming capability

## Directory Structure
```
project-root/
├── configs/
│   └── denseclip_cityscapes.yaml          # Main configuration file
├── data/
│   └── cityscapes/                        # Dataset directory
│       ├── leftImg8bit/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── gtFine/
│           ├── train/
│           ├── val/
│           └── test/
├── segmentation/
│   ├── pretrained/
│   │   ├── download_clip_models.sh        # Script to download CLIP weights
│   │   ├── RN50.pt                        # CLIP weights (after download)
│   │   ├── RN101.pt
│   │   └── ViT-B-16.pt
│   ├── train_denseclip.py                 # Main training script
│   └── ... (other model files)
├── work_dirs/                             # Training outputs
│   └── cityscapes_run_RN50_frozen/
│       ├── checkpoints/                   # Model checkpoints
│       │   ├── epoch_1.pth
│       │   └── ...
│       ├── logs/                          # Training logs
│       └── config.yaml                    # Saved config copy
└── README.md
```

## Installation

### Prerequisites
- Python 3.8
- CUDA-enabled GPU
- `git`, `wget`, `unzip`

### Setup
#### Clone the repository:
```bash
git clone <your-repository-url>
cd <your-repository-name>
```
#### Create and activate conda environment:
```bash
conda create -n denseclip_env python=3.8 -y
conda activate denseclip_env
```
#### Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml tqdm matplotlib opencv-python albumentations torchmetrics scipy Pillow wget
```

## Dataset Setup
#### Download Cityscapes dataset:
```bash
mkdir -p data
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<YourCityscapesUsername>&password=<YourCityscapesPassword>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -O data/leftImg8bit_trainvaltest.zip
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -O data/gtFine_trainvaltest.zip
rm cookies.txt
```
#### Extract the dataset:
```bash
unzip data/leftImg8bit_trainvaltest.zip -d data/cityscapes/
unzip data/gtFine_trainvaltest.zip -d data/cityscapes/
```

## Download Pre-trained Weights
```bash
cd segmentation/pretrained/
bash download_clip_models.sh
cd ../../
```

## Configuration
Edit `configs/denseclip_cityscapes.yaml` to ensure:
```yaml
data:
  path: 'data/cityscapes'  # Verify this path
  
model:
  clip_pretrained: 'segmentation/pretrained/ViT-B-16.pt'  # Path to CLIP weights
```

## Usage
### Training
```bash
python segmentation/train_denseclip.py configs/denseclip_cityscapes.yaml --work-dir work_dirs/cityscapes_run_RN50_frozen
```
### Resume Training
```bash
python segmentation/train_denseclip.py configs/denseclip_cityscapes.yaml \
    --work-dir work_dirs/cityscapes_run_RN50_frozen \
    --resume work_dirs/cityscapes_run_RN50_frozen/checkpoints/epoch_10.pth
