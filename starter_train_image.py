"""Train a binary image classifier: healthy vs damaged.
Run from the project root: python starter_train_image.py
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
SEED, EPOCHS, BATCH_SIZE, LR = 42, 12, 16, 1e-4
CLASSES = ["healthy", "damaged"]

class ImageDataset(Dataset):
    def __init__(self, frame, training=False):
        self.base = ROOT / "data/images"
        self.df = frame.reset_index(drop=True)
        ops = [transforms.Resize((224, 224))]
        if training:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize([0.485]*3, [0.229]*3)]
        self.tf = transforms.Compose(ops)

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.base / row.filename) as im:
            x = self.tf(im.convert("RGB"))
        return x, CLASSES.index(row["class"])

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ROOT / "data/images"
    frame = pd.read_csv(base / "labels.csv")
    frame = frame[frame["class"].isin(CLASSES)].copy()
    if frame.empty or frame["class"].value_counts().reindex(CLASSES, fill_value=0).min() < 2:
        raise ValueError("Image labels must contain at least two samples per class.")
    tr, va = train_test_split(frame, test_size=0.2, random_state=SEED, stratify=frame["class"])
    train_loader = DataLoader(ImageDataset(tr, True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ImageDataset(va, False), batch_size=BATCH_SIZE, shuffle=False)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES)); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(); best_f1 = -1; best_path = ROOT / "image_model.pth"
    for epoch in range(EPOCHS):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); opt.zero_grad(); logits = model(x); loss = loss_fn(logits, y); loss.backward(); opt.step(); running += loss.item()*len(y)
        model.eval(); yt=[]; yp=[]
        with torch.no_grad():
            for x, y in val_loader:
                yp.extend(model(x.to(device)).argmax(1).cpu().tolist()); yt.extend(y.tolist())
        score = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} train_loss={running/len(tr):.4f} val_accuracy={accuracy_score(yt,yp):.3f} val_macro_f1={score:.3f}")
        if score > best_f1:
            best_f1 = score
            torch.save({"state_dict": model.cpu().state_dict(), "classes": CLASSES, "seed": SEED, "val_macro_f1": score}, best_path)
            model.to(device)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval(); yt=[]; yp=[]
    with torch.no_grad():
        for x,y in val_loader:
            yp.extend(model(x.to(device)).argmax(1).cpu().tolist()); yt.extend(y.tolist())
    print(f"Best validation macro-F1: {checkpoint['val_macro_f1']:.3f}")
    print(classification_report(yt, yp, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):\n", confusion_matrix(yt, yp, labels=list(range(len(CLASSES)))))
    print("Saved:", best_path)


if __name__ == "__main__": 
    main()
