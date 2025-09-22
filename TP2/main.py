import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

hyper_params = {"backbone": "efficientnet_b4", "lr": 1e-4, "batch_size": 32, "epochs": 2}

# -----------------------------
# Load FairFace via Hugging Face datasets
# -----------------------------
ds = load_dataset("HuggingFaceM4/FairFace", "1.25")

train_ds_raw = ds['train']
val_ds_raw = ds['validation']

# -----------------------------
# Custom Dataset for PyTorch
# -----------------------------
from PIL import Image
import io

class FairFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img_bytes = row["image"]  # HF dataset returns bytes
        img = img_bytes.convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Age, gender, race are assumed integer labels in HF dataset
        age = int(row["age"])
        gender = int(row["gender"])
        race = int(row["race"])

        return img, (age, gender, race)

# -----------------------------
# Transforms & DataLoaders
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet input size
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])

train_ds = FairFaceDataset(train_ds_raw, transform=transform)
val_ds = FairFaceDataset(val_ds_raw, transform=transform)

train_loader = DataLoader(train_ds, batch_size=hyper_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=hyper_params["batch_size"], shuffle=False)

# -----------------------------
# EfficientNet Multi-Head Model
# -----------------------------
class MultiHeadEffNet(nn.Module):
    def __init__(self, backbone, n_age=100, n_gender=2, n_race=7):
        super().__init__()
        self.base = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.base.num_features

        self.fc_age = nn.Linear(feat_dim, n_age)
        self.fc_gender = nn.Linear(feat_dim, n_gender)
        self.fc_race = nn.Linear(feat_dim, n_race)

    def forward(self, x):
        f = self.base(x)
        age = self.fc_age(f)
        gender = self.fc_gender(f)
        race = self.fc_race(f)
        return age, gender, race

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadEffNet(hyper_params["backbone"]).to(device)

# -----------------------------
# Losses & Optimizer
# -----------------------------
criterion_age = nn.CrossEntropyLoss()
criterion_gender = nn.CrossEntropyLoss()
criterion_race = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyper_params["lr"])

# -----------------------------
# Training / Evaluation
# -----------------------------
def train_epoch(loader):
    model.train()
    total_loss = 0
    for imgs, (age, gender, race) in loader:
        imgs, age, gender, race = imgs.to(device), age.to(device), gender.to(device), race.to(device)

        optimizer.zero_grad()
        out_age, out_gender, out_race = model(imgs)

        loss = (criterion_age(out_age, age)
                + criterion_gender(out_gender, gender)
                + criterion_race(out_race, race))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader):
    model.eval()
    age_preds, age_labels = [], []
    with torch.no_grad():
        for imgs, (age, gender, race) in loader:
            imgs = imgs.to(device)
            out_age, out_gender, out_race = model(imgs)
            age_preds.extend(out_age.argmax(1).cpu().numpy())
            age_labels.extend(age.numpy())
    return accuracy_score(age_labels, age_preds)

# -----------------------------
# MLflow logging
# -----------------------------
mlflow.set_experiment("FairFace_EfficientNet_HF")

with mlflow.start_run():
    mlflow.log_params()

    for epoch in range(hyper_params["epochs"]):
        train_loss = train_epoch(train_loader)
        val_acc = eval_epoch(val_loader)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_acc_age", val_acc, step=epoch)

    # Log full model
    example_img, _ = next(iter(val_loader))
    example_img = example_img.to(device)
    mlflow.pytorch.log_model(model, "fairface_effnet_hf",
                             input_example=example_img,
                             registered_model_name="FairFaceEfficientNet_HF")
