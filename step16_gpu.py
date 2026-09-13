"""
Standalone GPU runner for notebook step 16 (multi-seed 2x2 grid).

Reconstructs, verbatim, the state that steps 1-9a build up inside the notebook
(transforms, datasets, the stratified split, loaders) so the 240/60 train/val
split and the augmented-training loader are byte-for-byte the same ones the
rest of the notebook's steps 6/9/10/11 were evaluated against. Runs the actual
20 seed x variant trainings on GPU, then dumps results as JSON so they can be
patched into the notebook's step-16 cell outputs without re-executing the
other 77 CPU-computed cells (which would otherwise re-roll their own
CPU floating-point non-determinism and require re-syncing all their prose).
"""
import copy
import json
import os
import random
import statistics
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy, F1Score
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# ---- cell f28be148 (imports + global seeds) --------------------------------
torch.manual_seed(101010)
np.random.seed(101010)
random.seed(101010)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

# ---- cell d9568f0d (unzip, if needed) ---------------------------------------
if not os.path.exists("data/chestxrays"):
    with zipfile.ZipFile("data/chestxrays.zip", "r") as zip_ref:
        zip_ref.extractall("data")

# ---- cell 23a95840 (transforms, base datasets/loaders) ----------------------
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std),
])

train_dataset = ImageFolder("data/chestxrays/train", transform=transform)
test_dataset = ImageFolder("data/chestxrays/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# ---- cell 12dc496c (step 5a: stratified train/val split) --------------------
val_fraction = 0.2
rng = np.random.default_rng(101010)

train_idx, val_idx = [], []
for cls in (0, 1):
    cls_idx = [i for i, t in enumerate(train_dataset.targets) if t == cls]
    cls_idx = rng.permutation(cls_idx)
    n_val = int(len(cls_idx) * val_fraction)
    val_idx.extend(cls_idx[:n_val])
    train_idx.extend(cls_idx[n_val:])

train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)

train_loader_s = DataLoader(train_subset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64)
print(f"train: {len(train_subset)}  val: {len(val_subset)}  steps/epoch: {len(train_loader_s)}")

# ---- cell 6e9f9696 (step 9a: augmented training loader) ---------------------
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std),
])
train_dataset_aug = ImageFolder("data/chestxrays/train", transform=augment_transform)
train_subset_aug = Subset(train_dataset_aug, train_idx)
train_loader_aug = DataLoader(train_subset_aug, batch_size=8, shuffle=True)
print(f"augmented train loader: {len(train_subset_aug)} images, {len(train_loader_aug)} steps/epoch")

# ---- cell 88e7e088 (pretrained weights cache location) ----------------------
os.environ["TORCH_HOME"] = "models"

# ---- cell 7afd1bf4 (make_model / evaluate_loss, now device-aware) -----------
criterion = nn.BCEWithLogitsLoss()


def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(device)


def evaluate_loss(m, loader):
    m.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = m(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
            n += inputs.size(0)
    return total_loss / n, correct / n


# ---- cell 16a (train_variant / evaluate_on_test, GPU-aware) -----------------
def train_variant(seed, loader, unfreeze_layer4, num_epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = make_model()

    if unfreeze_layer4:
        for p in m.layer4.parameters():
            p.requires_grad = True
        opt = optim.Adam([
            {"params": m.fc.parameters(), "lr": 0.001},
            {"params": m.layer4.parameters(), "lr": 0.0001},
        ])
    else:
        opt = optim.Adam(m.fc.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(num_epochs):
        m.train()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            opt.zero_grad()
            loss = criterion(m(inputs), labels)
            loss.backward()
            opt.step()

        val_loss, val_acc = evaluate_loss(m, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(m.state_dict())
            best_epoch = epoch + 1

    m.load_state_dict(best_state)
    return m, best_epoch, best_val_loss


accuracy_metric = Accuracy(task="binary")
f1_metric = F1Score(task="binary")


def evaluate_on_test(m):
    m.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = m(inputs)
            preds = torch.sigmoid(outputs).round().cpu()
            preds_list.extend(preds.tolist())
            labels_list.extend(labels.unsqueeze(1).tolist())
    preds_t = torch.tensor(preds_list)
    labels_t = torch.tensor(labels_list)
    acc = accuracy_metric(preds_t, labels_t).item()
    f1 = f1_metric(preds_t, labels_t).item()
    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    tn = ((preds_t == 0) & (labels_t == 0)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return dict(acc=acc, f1=f1, sens=sens, spec=spec)


# ---- cell 16b (the actual 5 seeds x 4 variants = 20 runs) -------------------
seeds = [101010, 202020, 303030, 404040, 505050]
variant_defs = [
    ("frozen + plain", train_loader_s, False),
    ("frozen + aug", train_loader_aug, False),
    ("layer4 + plain", train_loader_s, True),
    ("layer4 + aug", train_loader_aug, True),
]

results = []
for variant_name, loader, unfreeze in variant_defs:
    for seed in seeds:
        m, best_epoch, best_val_loss = train_variant(seed, loader, unfreeze)
        metrics = evaluate_on_test(m)
        metrics.update(variant=variant_name, seed=seed, best_epoch=best_epoch,
                        best_val_loss=best_val_loss)
        results.append(metrics)
        print(f"{variant_name:16s}  seed={seed}  best_epoch={best_epoch:2d}/25  "
              f"acc={metrics['acc']:.3f}  f1={metrics['f1']:.3f}  "
              f"sens={metrics['sens']:.3f}  spec={metrics['spec']:.3f}")

# ---- cell 16c (mean +/- std summary) -----------------------------------------
def summarize(key, rows):
    vals = [r[key] for r in rows]
    return statistics.mean(vals), statistics.stdev(vals)

summary = {}
print(f"\n{'variant':16s}  {'acc':>16s}  {'f1':>16s}  {'sens':>16s}  {'spec':>16s}")
for variant_name, _, _ in variant_defs:
    rows = [r for r in results if r["variant"] == variant_name]
    acc_m, acc_s = summarize("acc", rows)
    f1_m, f1_s = summarize("f1", rows)
    sens_m, sens_s = summarize("sens", rows)
    spec_m, spec_s = summarize("spec", rows)
    summary[variant_name] = dict(acc=(acc_m, acc_s), f1=(f1_m, f1_s),
                                  sens=(sens_m, sens_s), spec=(spec_m, spec_s))
    print(f"{variant_name:16s}  {acc_m:.3f} +/- {acc_s:.3f}  {f1_m:.3f} +/- {f1_s:.3f}  "
          f"{sens_m:.3f} +/- {sens_s:.3f}  {spec_m:.3f} +/- {spec_s:.3f}")

with open("step16_gpu_results.json", "w") as f:
    json.dump({"results": results, "summary": summary}, f, indent=2)
print("\nSaved step16_gpu_results.json")
