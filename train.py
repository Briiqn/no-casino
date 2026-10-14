import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt

from dataset import MrBeastScamDataset
from model import MrBeastScamClassifier

def smooth_labels(targets, factor=0.05):
    return targets * (1.0 - factor) + 0.5 * factor

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, targets)
                
            probs = torch.sigmoid(logits)
            
            total_loss += loss.item() * len(targets)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    avg_loss = total_loss / len(loader.dataset)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    
    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(all_targets, preds)),
        "precision": float(precision_score(all_targets, preds, zero_division=0)),
        "recall": float(recall_score(all_targets, preds, zero_division=0)),
        "f1": float(f1_score(all_targets, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_targets, all_probs)),
        "pr_auc": float(average_precision_score(all_targets, all_probs)),
    }
    return metrics, all_targets, all_probs

def train_model(
    backbone="convnext_tiny",
    epochs=25,
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-2,
    label_smoothing=0.05,
    save_path="best_mrbeast_scam_model.pth"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone: {backbone} | Batch size: {batch_size} | Epochs: {epochs}")
    print(f"Adversarial Defense Training: Memes, MrBeast Logos, Jimmy Poses, Elon/X Posts")
    print(f"==================================================")
    
    train_dataset = MrBeastScamDataset(split="train")
    val_dataset = MrBeastScamDataset(split="val")
    test_dataset = MrBeastScamDataset(split="test")
    
    print(f"Samples: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    batch_size = 32
    accumulation_steps = 2
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    model = MrBeastScamClassifier(backbone=backbone, pretrained=True, dropout=0.4).to(device)
    criterion_train = nn.BCEWithLogitsLoss()
    criterion_eval = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "val_auc": []}
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            smoothed_targets = smooth_labels(targets, factor=label_smoothing)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion_train(logits, smoothed_targets) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            train_loss += (loss.item() * accumulation_steps) * len(targets)
            
        scheduler.step()
        train_loss = train_loss / len(train_dataset)
        
        val_metrics, _, _ = evaluate(model, val_loader, criterion_eval, device)
        epoch_time = time.time() - epoch_start
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["roc_auc"])
        
        is_best = False
        if val_metrics["f1"] >= best_val_f1 and val_metrics["loss"] < best_val_loss:
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]
            is_best = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": backbone,
                "val_metrics": val_metrics,
            }, save_path)
            
        best_marker = "  [★ BEST]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
              f"Train: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}{best_marker}")
            
    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins)!")
    
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val F1: {checkpoint['val_metrics']['f1']:.4f}")
    
    print("\nEvaluating best model on held-out TEST set (1,600 images)...")
    test_metrics, test_targets, test_probs = evaluate(model, test_loader, criterion_eval, device)
    
    print("---------------- TEST METRICS ----------------")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
        
    cm = confusion_matrix(test_targets, (test_probs >= 0.5).astype(int))
    print("\nConfusion Matrix (Test):")
    print(f"  [TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"  [FN={cm[1,0]}  TP={cm[1,1]}]")
    
    results = {
        "backbone": backbone,
        "epochs": epochs,
        "best_epoch": checkpoint["epoch"],
        "total_training_time_sec": total_training_time,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "history": history
    }
    with open("training_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="crimson")
    axes[0].set_title("Adversarial Defense Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {test_metrics['roc_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[1].set_title("Test ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True)
    
    im = axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[2].set_title("Test Confusion Matrix (1,600 images)")
    fig.colorbar(im, ax=axes[2])
    classes = ["Legit/Adv (0)", "Scam (1)"]
    tick_marks = np.arange(len(classes))
    axes[2].set_xticks(tick_marks)
    axes[2].set_xticklabels(classes)
    axes[2].set_yticks(tick_marks)
    axes[2].set_yticklabels(classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=14)
    axes[2].set_ylabel("True Label")
    axes[2].set_xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)
    plt.close()
    print("Metrics and plots saved successfully!")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_model(epochs=20, batch_size=64)
