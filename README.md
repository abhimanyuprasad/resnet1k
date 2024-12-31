# ResNet50 Training on ImageNet-1K

This repository contains an implementation of ResNet50 trained from scratch on the ImageNet-1K dataset. The implementation includes data loading, model architecture, training pipeline, and model checkpointing.

## Project Structure 

├── train_resnet.py # Main training script
├── download_tiny_imagenet.py # Script to download Tiny-ImageNet
├── download_imagenet_torrent.py # Script to download ImageNet-1K

└── models/ # Directory for saved models

├── resnet50_imagenet1k_best.pth
├── resnet50_imagenet1k_final.pth
└── resnet50_imagenet1k_epoch_.pth

## Requirements
pip install torch torchvision tqdm requests scipy

## Model Architecture

The ResNet50 implementation includes:
- 50 layers deep with residual connections
- Bottleneck blocks for efficient training
- Batch normalization after each convolution
- Final 1000-way classification layer
- Kaiming initialization for weights

Key components:
- Initial 7x7 convolution with stride 2
- Max pooling with stride 2
- 4 stages of residual blocks
- Global average pooling
- Final fully connected layer

## Dataset

### ImageNet-1K
- 1000 classes
- ~1.2 million training images
- 50,000 validation images
- Images resized to 224x224 for training

Directory structure:
imagenet1k/
train/
n01440764/
n01443537/
...
val/
n01440764/
n01443537/
...

## Training Configuration

- Batch size: 256
- Learning rate: 0.1 with cosine annealing
- Weight decay: 1e-4
- Momentum: 0.9
- Mixed precision training
- Learning rate warmup for 5 epochs

Data augmentation:
- Random resized crop to 224x224
- Random horizontal flip
- Normalization with ImageNet statistics

## Usage

1. Download ImageNet-1K dataset:
bash
python download_imagenet_torrent.py

2. Train the model:
bash
python train_resnet.py

3. Load a trained model:
python
import torch
from train_resnet import ResNet50
Load model
model = ResNet50(num_classes=1000)
checkpoint = torch.load('models/resnet50_imagenet1k_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

## Model Checkpoints

The training script saves several checkpoints:
- `resnet50_imagenet1k_best.pth`: Best model based on validation accuracy
- `resnet50_imagenet1k_final.pth`: Final model after training completion
- `resnet50_imagenet1k_epoch_*.pth`: Periodic checkpoints every 10 epochs

Checkpoint contents:

python
{
'epoch': epoch_number,
'model_state_dict': model_state,
'optimizer_state_dict': optimizer_state,
'scheduler_state_dict': scheduler_state,
'val_acc': validation_accuracy,
'val_loss': validation_loss,
'train_acc': training_accuracy,
'train_loss': training_loss
}

## Performance Monitoring

The training script outputs:
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Learning rate updates
- Best model validation accuracy
- Progress bars with real-time metrics

## Implementation Details

### ResBlock (Bottleneck Block)
python
1x1 conv, reduce dimensions
3x3 conv, process features
1x1 conv, restore dimensions
Skip connection

### Learning Rate Schedule
- Warmup for first 5 epochs
- Cosine annealing for remaining epochs

### Optimization
- SGD optimizer with momentum
- Mixed precision training for efficiency
- Gradient scaling for numerical stability