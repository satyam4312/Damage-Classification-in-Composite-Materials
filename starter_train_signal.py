"""Leakage-aware, regularized signal classifier.
Run from damage_detection_updated/: python starter_train_signal.py

Important: filenames encode class names, but filenames are never model inputs. The
provided dataset has no specimen/session IDs, so this is a sample-level split; it
cannot establish specimen-independent generalization.
"""
from pathlib import Path
import random, copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
CLASSES = ["healthy", "matrix_crack", "delamination", "fiber_break"]
SEED, EPOCHS, BATCH_SIZE, LR, PATIENCE = 42, 30, 16, 2e-5, 6


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)


class SignalDataset(Dataset):
    def __init__(self, frame, training=False):
        self.base = ROOT / "data/signals"
        self.df = frame.reset_index(drop=True)
        self.training = training
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ])

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = pd.read_csv(self.base / row.filename)["amplitude"].to_numpy(dtype=np.float32)
        if len(arr) < 16 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid or too-short amplitude data in {row.filename}")
        # Robust per-record centering/scaling reduces dependence on absolute gain.
        arr = arr - np.median(arr)
        scale = np.percentile(np.abs(arr), 95) + 1e-8
        arr = np.clip(arr / scale, -5, 5)
        # Training-only signal-domain augmentation: mild gain, noise and circular shift.
        if self.training:
            arr = arr * np.random.uniform(0.8, 1.2)
            arr = np.roll(arr, np.random.randint(0, max(1, min(32, len(arr)))))
            arr = arr + np.random.normal(0, 0.015, size=arr.shape).astype(np.float32)
            if np.random.rand() < 0.25:
                mask_len = max(1, int(len(arr) * np.random.uniform(0.005, 0.02)))
                start = np.random.randint(0, max(1, len(arr)-mask_len+1))
                arr[start:start+mask_len] = 0
        # Use fixed, documented assumed sampling rate; update if metadata provides real rate.
        _, _, power = spectrogram(arr, fs=2048, nperseg=min(256, len(arr)), noverlap=min(192, max(0, min(256, len(arr))-1)))
        spec = np.log1p(power).astype(np.float32)
        lo, hi = np.percentile(spec, [1, 99])
        spec = np.clip((spec-lo)/(hi-lo+1e-8), 0, 1)
        img = Image.fromarray(np.uint8(spec * 255)).convert("RGB")
        return self.tf(img), CLASSES.index(row["class"])


def make_loaders(frame, device):
    # Separate locked test set before any model selection; split indices only.
    trainval, test = train_test_split(frame, test_size=0.15, random_state=SEED,
                                      stratify=frame["class"])
    train, val = train_test_split(trainval, test_size=0.1764705882, random_state=SEED,
                                  stratify=trainval["class"])  # 70/15/15 overall
    pin = device.type == "cuda"
    loaders = {
        "train": DataLoader(SignalDataset(train, True), batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=pin),
        "val": DataLoader(SignalDataset(val, False), batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=pin),
        "test": DataLoader(SignalDataset(test, False), batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=pin),
    }
    print(f"Split sizes: train={len(train)}, validation={len(val)}, locked_test={len(test)}")
    print("WARNING: no specimen/session ID was found; these are sample-level splits.")
    return loaders


def evaluate(model, loader, device):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().tolist()
            yp.extend(pred); yt.extend(y.tolist())
    return yt, yp


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(ROOT / "data/signals/labels.csv")
    required = {"filename", "class"}
    if not required.issubset(frame.columns):
        raise ValueError(f"labels.csv must contain {sorted(required)}; found {list(frame.columns)}")
    frame = frame[frame["class"].isin(CLASSES)].copy()
    frame["filename"] = frame["filename"].astype(str)
    frame = frame.drop_duplicates(subset=["filename"])
    missing = [f for f in frame.filename if not (ROOT / "data/signals" / f).is_file()]
    if missing: raise FileNotFoundError(f"Missing signal files, e.g. {missing[:5]}")
    counts = frame["class"].value_counts().reindex(CLASSES, fill_value=0)
    if (counts < 3).any(): raise ValueError(f"Need at least 3 examples/class: {counts.to_dict()}")
    print("Class counts:\n", counts.to_string())
    loaders = make_loaders(frame, device)

    # Smaller regularized head and frozen early backbone reduce overfitting on 300 samples.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, len(CLASSES)))
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LR, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    best_f1, stale, best_state = -1.0, 0, None
    out_path = ROOT / "signal_model.pth"

    for epoch in range(EPOCHS):
        model.train(); running = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); running += loss.item() * len(y)
        yt, yp = evaluate(model, loaders["val"], device)
        val_f1 = f1_score(yt, yp, average="macro", zero_division=0)
        val_acc = accuracy_score(yt, yp)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} train_loss={running/len(loaders['train'].dataset):.4f} "
              f"val_accuracy={val_acc:.3f} val_macro_f1={val_f1:.3f}")
        if val_f1 > best_f1 + 1e-4:
            best_f1, stale = val_f1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save({"state_dict": best_state, "classes": CLASSES, "seed": SEED,
                        "best_val_macro_f1": best_f1, "sampling_rate_hz_assumed": 2048,
                        "input_type": "log-power spectrogram", "normalization": "per-record robust percentile",
                        "split": "70/15/15 sample-level stratified; no specimen IDs available"}, out_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping after {epoch+1} epochs."); break

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    yt, yp = evaluate(model, loaders["test"], device)
    print(f"Best validation macro-F1: {checkpoint['best_val_macro_f1']:.3f}")
    print(f"LOCKED TEST accuracy={accuracy_score(yt, yp):.3f}; macro-F1={f1_score(yt, yp, average='macro', zero_division=0):.3f}")
    print("Test classification report:\n", classification_report(yt, yp, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0))
    print("Test confusion matrix (rows=true, columns=predicted):\n", confusion_matrix(yt, yp, labels=list(range(len(CLASSES)))))
    print("Saved:", out_path)

if __name__ == "__main__": 
    main()
