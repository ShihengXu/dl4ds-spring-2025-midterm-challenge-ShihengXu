import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb

################################################################################
# ✅ Model Definition - ResNet50 (Predefined Model)
################################################################################
class CustomResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomResNet, self).__init__()
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)  # Modify final layer
    
    def forward(self, x):
        return self.model(x)

################################################################################
# ✅ Training Function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train for one epoch."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({"loss": running_loss / (total // labels.size(0)), "acc": 100. * correct / total})
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

################################################################################
# ✅ Validation Function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({"loss": running_loss / (total // labels.size(0)), "acc": 100. * correct / total})
    
    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

################################################################################
# ✅ Main Execution
################################################################################
def main():
    CONFIG = {
        "model": "ResNet50",
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 20,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }
    
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    device = CONFIG["device"]
    print(f"🚀 Using device: {device}")
    
    # ✅ Define Data Augmentation and Normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # ✅ Load CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # ✅ Initialize Model, Loss, and Optimizer
    model = CustomResNet(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    
    # ✅ Training Loop
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        
        # ✅ Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        
        # ✅ Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model_part2.pth")
            wandb.save("model_part2.pth")
    
    wandb.finish()
    print(f"✅ Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
