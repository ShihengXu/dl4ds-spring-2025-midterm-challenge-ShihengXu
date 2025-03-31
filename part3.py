import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json
import torch.cuda.amp as amp  # ✅ Mixed Precision
from torch.optim.swa_utils import AveragedModel, SWALR  # ✅ SWA

################################################################################
# ✅ Force GPU Usage and Optimized Configurations
################################################################################
CONFIG = {
    "model": "EfficientNet-B3_Finetuned",
    "batch_size": 128,  # Keep batch size optimized for 4090
    "learning_rate": 3e-4,  # Lower LR helps fine-tune EfficientNet
    "epochs": 100,  # Increase training time
    "num_workers": 12,  # Optimize for NVIDIA RTX 4090
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",
    "ood_dir": "./data/ood-test",
    "wandb_project": "sp25-ds542-challenge",
    "seed": 42,
}

DEVICE = CONFIG["device"]

################################################################################
# ✅ Improved Model (Using Pretrained EfficientNet-B3)
################################################################################
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomEfficientNet, self).__init__()
        self.model = models.efficientnet_b3(weights="IMAGENET1K_V1")  # Load Pretrained Model
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),  # ✅ Add Dropout for regularization
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

################################################################################
# ✅ Training Function (Uses Mixed Precision, SWA, and EMA)
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, scaler, swa_model, ema_model):
    """Train one epoch using Mixed Precision (FP16)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):  # ✅ Fixed

            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # ✅ Update SWA and EMA models
        if epoch >= 40:  # Apply SWA only in last 10 epochs
            swa_model.update_parameters(model)
            ema_model.update_parameters(model)

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    return running_loss / len(trainloader), 100. * correct / total

################################################################################
# ✅ Main Training Execution
################################################################################
def main():
    """Main training script with enforced GPU usage."""

    # ✅ Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected! Running on CPU.")

    # ✅ Load Data
    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),  # ✅ Use RandAugment
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]))

    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # ✅ Initialize Model & Move to GPU
    model = CustomEfficientNet(num_classes=100).to(DEVICE)
    swa_model = AveragedModel(model)  # ✅ SWA
    ema_model = AveragedModel(model, avg_fn=lambda avg, new, _: 0.9 * avg + 0.1 * new)  # ✅ EMA

    # ✅ Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ✅ Label Smoothing
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)

    # ✅ Fix: Add 'swa_lr' to SWALR
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_strategy="cos", last_epoch=-1)

    # ✅ Mixed Precision
    scaler = torch.amp.GradScaler("cuda")  # ✅ Fixed

    # ✅ Train Model
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, scaler, swa_model, ema_model)

        # ✅ Apply SWA after epoch 40
        if epoch >= 40:
            swa_scheduler.step()

        # ✅ Save Best Model
        if train_acc > best_val_acc:
            best_val_acc = train_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"✅ Training Complete! Best Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()