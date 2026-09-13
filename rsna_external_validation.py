"""
External validation of notebook step 5/6's adopted model (frozen ResNet-18
head, trained on Kermany et al. 2018 pediatric AP chest X-rays) against the
RSNA Pneumonia Detection Challenge dataset (Shih et al. 2019) — a genuinely
different institution, population, and label definition.

Requires a free Kaggle account that has joined the RSNA Pneumonia Detection
Challenge competition (accept its rules at kaggle.com/c/rsna-pneumonia-
detection-challenge/rules), and a Kaggle API token saved to
~/.kaggle/access_token (kagglehub's token-auth file) or exported as the
KAGGLE_API_TOKEN environment variable. See kagglehub's README for details.

Citation required by RSNA's data-use terms for any use of this data:
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-
  scale Chest X-ray Database and Benchmarks on Weakly-Supervised
  Classification and Localization of Common Thorax Diseases. CVPR 2017.
- Shih G, Wu CC, Halabi SS, et al. Augmenting the National Institutes of
  Health Chest Radiograph Dataset with Expert Annotations of Possible
  Pneumonia. Radiology: Artificial Intelligence 2019;1(1):e180041.

Design notes (see notebook step 17 and chat history for full reasoning):
- RSNA's 3-way label ("Normal" / "Lung Opacity" / "No Lung Opacity-Not
  Normal") is collapsed to a 2-way NORMAL/PNEUMONIA-analog comparison by
  EXCLUDING "No Lung Opacity / Not Normal" (11,821 images) — it doesn't map
  cleanly to either of this project's two classes.
- Uses the FULL usable set (14,863 images), not a subsample: GPU inference
  is cheap (forward pass only), and a larger n gives a much tighter external
  CI than this notebook's own internal n=100 evaluation set.
- Reports results POOLED and STRATIFIED BY ViewPosition (AP/PA). This
  matters because RSNA's own AP/PA split is strongly confounded with the
  label (Lung Opacity is ~78% AP; Normal is ~81% PA — sicker patients get
  portable/AP films in practice) while the training data (Kermany) is
  AP-only — a pooled number alone would conflate "different population"
  with "unfamiliar projection type."
- Also reports each stratum's own trivial (prevalence-matched) baseline
  accuracy for comparison, in the same spirit as the notebook's own
  trivial-baseline check after step 6.
"""
import os

import kagglehub
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ---- 1. Download (cached after first run) -----------------------------------
BASE = kagglehub.competition_download("rsna-pneumonia-detection-challenge")
IMG_DIR = os.path.join(BASE, "stage_2_train_images")
print(f"data at: {BASE}")

# ---- 2. Build a manifest: labels + class + DICOM ViewPosition/Age/Sex ------
manifest_path = os.path.join(BASE, "manifest.csv")
if os.path.exists(manifest_path):
    manifest = pd.read_csv(manifest_path)
else:
    labels = pd.read_csv(os.path.join(BASE, "stage_2_train_labels.csv"))
    labels = labels.drop_duplicates("patientId")[["patientId", "Target"]]
    classes = pd.read_csv(os.path.join(BASE, "stage_2_detailed_class_info.csv"))
    classes = classes.drop_duplicates("patientId")
    meta = labels.merge(classes, on="patientId", how="left")

    rows = []
    for pid in meta["patientId"]:
        ds = pydicom.dcmread(os.path.join(IMG_DIR, f"{pid}.dcm"), stop_before_pixels=True)
        rows.append({
            "patientId": pid,
            "ViewPosition": getattr(ds, "ViewPosition", None),
            "PatientAge": getattr(ds, "PatientAge", None),
            "PatientSex": getattr(ds, "PatientSex", None),
        })
    manifest = meta.merge(pd.DataFrame(rows), on="patientId", how="left")
    manifest.to_csv(manifest_path, index=False)

usable = manifest[manifest["class"] != "No Lung Opacity / Not Normal"].copy()
usable["label"] = (usable["class"] == "Lung Opacity").astype(int)  # 1 = PNEUMONIA-analog
print(f"usable images: {len(usable)}  (Normal={sum(usable.label == 0)}, Lung Opacity={sum(usable.label == 1)})")
print(usable.groupby(["ViewPosition", "label"]).size())

# ---- 3. Preprocessing: DICOM -> 3-channel 224x224, ImageNet-normalized -----
# Same normalization constants as the main notebook (cell 23a95840). Unlike
# Kermany's pre-sized images, RSNA's 1024x1024 DICOMs need an explicit resize.
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
rsna_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std),
])


class RSNADataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ds = pydicom.dcmread(os.path.join(self.img_dir, f"{row.patientId}.dcm"))
        img = Image.fromarray(ds.pixel_array).convert("RGB")  # replicate to 3ch, like Kermany's JPEGs
        return self.transform(img), row.label


loader = DataLoader(RSNADataset(usable, IMG_DIR, rsna_transform), batch_size=128,
                     shuffle=False, num_workers=0)  # num_workers=0: avoids a Docker /dev/shm bus error

# ---- 4. Load the step 5/6 model (full state: backbone + trained head) -----
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("models/resnet18_pneumonia_full_state.pt", map_location=device))
model = model.to(device).eval()
print("loaded models/resnet18_pneumonia_full_state.pt (step 5/6's adopted model)")

# ---- 5. Inference over all 14,863 images -----------------------------------
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels_batch in loader:
        outputs = model(inputs.to(device))
        all_preds.extend(torch.sigmoid(outputs).round().squeeze(1).cpu().tolist())
        all_labels.extend(labels_batch.tolist())

usable["pred"] = all_preds
usable["label"] = all_labels
usable.to_csv(os.path.join(BASE, "external_validation_predictions.csv"), index=False)


# ---- 6. Report: pooled, stratified by ViewPosition, vs. trivial baselines --
def confusion_report(df, name):
    labels_t = torch.tensor(df["label"].values)
    preds_t = torch.tensor(df["pred"].values)
    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    tn = ((preds_t == 0) & (labels_t == 0)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()
    n = len(df)
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
    always_pos_acc = (tp + fn) / n
    always_neg_acc = (tn + fp) / n
    beats_trivial = acc > max(always_pos_acc, always_neg_acc)
    print(f"\n{name}  (n={n})")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  accuracy={acc:.3f}  f1={f1:.3f}  sensitivity={sens:.3f}  specificity={spec:.3f}  precision={prec:.3f}")
    print(f"  trivial baselines: always-positive={always_pos_acc:.3f}  always-negative={always_neg_acc:.3f}"
          f"  -> model beats best trivial: {beats_trivial}")
    return dict(name=name, n=n, tn=tn, fp=fp, fn=fn, tp=tp, acc=acc, f1=f1, sens=sens, spec=spec,
                prec=prec, always_pos_acc=always_pos_acc, always_neg_acc=always_neg_acc,
                beats_trivial=beats_trivial)


results = [confusion_report(usable, "POOLED (all views)")]
for view in ["AP", "PA"]:
    results.append(confusion_report(usable[usable["ViewPosition"] == view], f"ViewPosition = {view}"))

pd.DataFrame(results).to_csv(os.path.join(BASE, "external_validation_results.csv"), index=False)
print(f"\nSaved results to {BASE}/external_validation_results.csv")
