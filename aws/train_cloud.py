# aws/train_cloud.py
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

print("--- Initializing Cloud Training Pipeline on AWS ---")

# 1. Dataset Configuration for Cloud Paths
data_dir = os.environ.get("DATA_DIR", "../COBRE_Dataset")
csv_path = os.path.join(data_dir, "participants.tsv")
df_participants = pd.read_csv(csv_path, sep='\t')

all_paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "anat", "*_T1w.nii.gz")))
path_dict = {os.path.basename(p).split('_')[0]: p for p in all_paths}

valid_participants = df_participants[df_participants['participant_id'].isin(path_dict.keys())].copy()
matched_paths = [path_dict[pid] for pid in valid_participants['participant_id']]

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(valid_participants['diagnosis'])
num_classes = len(encoder.classes_)

# 2. Compute Class Weights to fix the "Control Collapse"
class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

X_train_p, X_val_p, y_train_l, y_val_l = train_test_split(
    matched_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

class COBRE_AxialDataset(Dataset):
    def __init__(self, subject_paths, labels, img_size=128):
        self.subject_paths = subject_paths
        self.labels = labels
        self.img_size = img_size
    def __len__(self):
        return len(self.subject_paths)
    def __getitem__(self, idx):
        nii_img = nib.load(self.subject_paths[idx])
        volume = nii_img.get_fdata()
        slice_2d = volume[:, :, volume.shape[2] // 2]
        min_val, max_val = np.min(slice_2d), np.max(slice_2d)
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val)
        else:
            slice_2d = np.zeros_like(slice_2d)
        tensor_slice = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = nn.functional.interpolate(tensor_slice, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return resized.squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

train_loader = DataLoader(COBRE_AxialDataset(X_train_p, y_train_l), batch_size=16, shuffle=True)
val_loader = DataLoader(COBRE_AxialDataset(X_val_p, y_val_l), batch_size=16, shuffle=False)

# 3. Robust CNN Architecture
class SimpleBrainCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBrainCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = SimpleBrainCNN(num_classes=num_classes).to(device)
class_weights_tensor = class_weights_tensor.to(device)

# Pass class weights to CrossEntropyLoss to penalize ignoring minority classes!
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 4. Extended Cloud Training Loop (e.g., 15 epochs with GPU power)
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader.dataset):.4f} | Train Acc: {correct/total*100:.2f}%")

print("\nCloud Training Complete! Model weights ready for evaluation.")