import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import urllib.request
import torchvision.models as models

# --------------------------
# Model Definition (Must Match Training)
# --------------------------
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomEfficientNet, self).__init__()
        self.model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# --------------------------
# OOD Evaluation Functions
# --------------------------
def evaluate_ood(model, distortion_name, severity, CONFIG):
    """Evaluates the model on a specific OOD distortion level."""
    data_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]

    # ✅ Load distortion dataset
    file_path = os.path.join(data_dir, f"{distortion_name}.npy")
    images = np.load(file_path)

    # ✅ Extract severity-specific images
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # ✅ Convert to PyTorch tensors (Normalize like training data)
    images = torch.from_numpy(images).float() / 255.
    images = images.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)

    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"],
                                             shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    predictions = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
            inputs = batch[0].to(device)
            inputs = normalize(inputs)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())

    return predictions

def files_already_downloaded(directory, num_files):
    """Checks if all required OOD files are already downloaded."""
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return False
    return True

def download_ood_files(data_dir, num_files):
    """Downloads OOD dataset files if they are not already present."""
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"
    
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_url = base_url + file_name
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"✅ Downloaded {file_name}.")
        else:
            print(f"✔ {file_name} already exists.")

def evaluate_ood_test(model, CONFIG):
    """Evaluates model performance on all OOD distortions and severities."""
    data_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]
    
    num_files = 19  # Total distortion files: distortion00.npy ... distortion18.npy
    
    if not files_already_downloaded(data_dir, num_files):
        download_ood_files(data_dir, num_files)
    else:
        print("✅ OOD files already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(num_files)]
    
    all_predictions = []
    model.eval()
    
    for distortion in distortions:
        for severity in range(1, 6):
            preds = evaluate_ood(model, distortion, severity, CONFIG)
            all_predictions.extend(preds)
            print(f"✅ Evaluated {distortion} Severity {severity}")

    return all_predictions

def create_ood_df(all_predictions):
    """Creates a CSV file with OOD predictions."""
    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    
    ids_ood = []
    for distortion in distortions:
        for severity in range(1, 6):
            for i in range(10000):
                ids_ood.append(f"{distortion}_{severity}_{i}")

    submission_df_ood = pd.DataFrame({'id': ids_ood, 'label': all_predictions})
    return submission_df_ood

# --------------------------
# Main OOD Evaluation Execution
# --------------------------
def main():
    """Runs the OOD evaluation and saves predictions as submission_ood.csv."""
    CONFIG = {
        "batch_size": 128,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ood_dir": "./data/ood-test",
    }

    # ✅ Load the best model for OOD evaluation
    model = CustomEfficientNet(num_classes=100)
    model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    model.eval()
    
    print("🚀 Running OOD evaluation...")

    # ✅ Evaluate on OOD dataset
    all_predictions = evaluate_ood_test(model, CONFIG)
    
    # ✅ Create and save submission file
    submission_df = create_ood_df(all_predictions)
    submission_df.to_csv("submission_ood.csv", index=False)
    
    print("🎯 OOD evaluation complete. Submission saved to **submission_ood.csv**")

if __name__ == '__main__':
    main()
