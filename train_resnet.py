import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import os
import requests
import tarfile
from tqdm import tqdm
import shutil
from pathlib import Path
import scipy.io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for training data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Define transforms for validation data
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load Tiny-ImageNet dataset
def load_tiny_imagenet(root_dir):
    """Load Tiny-ImageNet dataset"""
    train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
    val_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val')
    
    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_transforms
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        transform=val_transforms
    )
    
    return train_dataset, val_dataset

# Create data loaders
def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Custom ResNet50 implementation
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels * 4, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Modify the dataset loading function
def load_imagenet1k(root_dir):
    """Load ImageNet-1K dataset"""
    print("Loading training data...")
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, 'train'),
        transform=train_transforms
    )
    
    print("Loading validation data...")
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, 'val'),
        transform=val_transforms
    )
    
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    return train_dataset, val_dataset

# Modify the load_model function to create ResNet50 from scratch
def load_model():
    """Create ResNet50 model from scratch"""
    model = ResNet50(num_classes=1000)  # ImageNet-1K has 1000 classes
    return model.to(device)

def ensure_model_dir():
    """Create model directory if it doesn't exist"""
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# Add this function after the imports
def reorganize_validation_data(root_dir):
    """
    Reorganize validation data to match training data structure.
    Uses training directory structure as reference.
    """
    val_dir = os.path.join(root_dir, 'val')
    train_dir = os.path.join(root_dir, 'train')
    
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")
    
    # Check if already organized
    if any(os.path.isdir(os.path.join(val_dir, d)) for d in os.listdir(val_dir)):
        print("Validation data already organized")
        return
    
    print("Reorganizing validation data...")
    
    # Get class folders from training directory
    class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not class_dirs:
        raise FileNotFoundError("No class directories found in training directory")
    
    print(f"Found {len(class_dirs)} classes")
    
    # Create temporary directory
    temp_dir = os.path.join(val_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move all validation images to temp directory
    print("Moving files to temporary directory...")
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.JPEG')]
    for img in tqdm(val_images):
        shutil.move(os.path.join(val_dir, img), os.path.join(temp_dir, img))
    
    # Create class directories and distribute images
    print("Creating class directories...")
    images_per_class = len(val_images) // len(class_dirs)
    
    for i, class_dir in enumerate(tqdm(class_dirs)):
        # Create class directory in validation
        class_path = os.path.join(val_dir, class_dir)
        os.makedirs(class_path, exist_ok=True)
        
        # Move corresponding images
        start_idx = i * images_per_class + 1
        end_idx = (i + 1) * images_per_class + 1
        
        for j in range(start_idx, end_idx):
            img_name = f'ILSVRC2012_val_{j:08d}.JPEG'
            if os.path.exists(os.path.join(temp_dir, img_name)):
                shutil.move(
                    os.path.join(temp_dir, img_name),
                    os.path.join(class_path, img_name)
                )
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("Validation data reorganization complete")

# Update the main function for ImageNet-1K training
def main():
    root_dir = '/opt/dlami/nvme/resnet50/imagenet1k/ILSVRC/Data/CLS-LOC'  # Change this to your ImageNet-1K directory
    batch_size = 256  # Larger batch size for ImageNet
    num_epochs = 6   # Standard number of epochs for ImageNet training
    
    # Reorganize validation data if needed
    print("Checking validation data structure...")
    reorganize_validation_data(root_dir)
    
    # Create model directory
    model_dir = ensure_model_dir()
    
    print("Load ImageNet-1K")
    train_dataset, val_dataset = load_imagenet1k(root_dir)
    
    print("Create data loader")
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        batch_size=batch_size
    )
    
    # Create new model
    model = load_model()
    
    # Use mixed precision training for better performance
    scaler = torch.cuda.amp.GradScaler()
    
    # Train model with mixed precision
    print("Train Model")
    train_model(model, train_loader, val_loader, num_epochs, scaler)
    
    # Save final model with metadata
    print("create final model path")
    final_model_path = os.path.join(model_dir, 'resnet50_imagenet1k_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 1000,
            'dataset': 'ImageNet-1K',
            'architecture': 'ResNet50',
            'input_size': 224,
            'num_params': sum(p.numel() for p in model.parameters()),
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'optimizer': 'SGD',
            'initial_lr': 0.1,
            'weight_decay': 1e-4,
        }
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

# Update train_model function for better training on ImageNet-1K
def train_model(model, train_loader, val_loader, num_epochs=90, scaler=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate schedule: warmup for 5 epochs and then cosine decay
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    
    best_val_acc = 0.0
    model_dir = ensure_model_dir()
    
    for epoch in range(num_epochs):
        # Warmup learning rate
        if epoch < warmup_epochs:
            lr = 0.1 * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Update learning rate
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
        }
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_dir, f'resnet50_imagenet1k_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(model_dir, 'resnet50_imagenet1k_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f"\nSaved best model with validation accuracy: {val_acc:.2f}%")
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('--------------------')

# Add validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

if __name__ == '__main__':
    main() 