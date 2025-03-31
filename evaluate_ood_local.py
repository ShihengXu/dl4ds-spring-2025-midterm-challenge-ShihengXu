import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel

# ✅ Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_part3_swa_mixup.pth"  # Change if needed
OOD_DIR = "./data/ood-test"
BATCH_SIZE = 128
NUM_WORKERS = 0  # 🔥 Set to 0 to disable multiprocessing on Windows!

# ✅ Normalize Transformation
normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

def evaluate_ood(model, distortion_name):
    """Evaluates the model on a specific OOD distortion."""
    file_path = os.path.join(OOD_DIR, f"{distortion_name}.npy")

    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        return None

    # ✅ Load OOD images
    images = np.load(file_path)
    images = torch.from_numpy(images).float() / 255.  # Normalize to [0,1]
    images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ✅ Compute confidence & entropy
    confidences = []
    entropies = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {distortion_name}", leave=False):
            inputs = batch[0].to(DEVICE)
            inputs = normalize(inputs)  # Apply normalization

            outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            conf, _ = torch.max(outputs, dim=1)
            entropy = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)

            confidences.extend(conf.cpu().numpy())
            entropies.extend(entropy.cpu().numpy())

    return np.mean(confidences), np.mean(entropies)

def main():
    print("🚀 Loading trained SWA model...")

    # ✅ Load ResNet-101 architecture
    model = torchvision.models.resnet101(weights=None)  # Initialize model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)  # CIFAR-100 has 100 classes

    # ✅ Wrap model with AveragedModel for SWA compatibility
    model = AveragedModel(model)

    # ✅ Load SWA-trained weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading SWA model: {e}")
        exit(1)

    model = model.to(DEVICE)
    model.eval()

    # ✅ Get list of distortions
    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    # ✅ Evaluate OOD distortions
    ood_results = []
    print("📊 Evaluating OOD Distortions...")
    for distortion in distortions:
        avg_conf, avg_entropy = evaluate_ood(model, distortion)
        if avg_conf is not None:
            ood_results.append((distortion, avg_conf, avg_entropy))

    # ✅ Print results
    print("\n📊 **OOD Evaluation Summary**")
    print(f"{'Distortion':<20} {'Avg Confidence':<20} {'Avg Entropy'}")
    print("=" * 60)
    for distortion, conf, entropy in ood_results:
        print(f"{distortion:<20} {conf:.4f} {entropy:.4f}")

    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
