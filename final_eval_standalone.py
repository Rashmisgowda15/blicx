import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)

# Constants
DATA_DIR = r"c:\Users\Rashmi Mithun\Desktop\new dental pro\data\classification_setup"
MODEL_PATH = "dental_cnn_model.pth"
BATCH_SIZE = 16
NUM_CLASSES = 2

def evaluate_standalone(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": auc(*roc_curve(y_true, y_prob)[:2])
    }

    print("\n--- FINAL EVALUATION RESULTS ---")
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    
    # Save the plots silently
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Final Confusion Matrix")
    plt.savefig('final_confusion_matrix.png')
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    plt.plot([0,1],[0,1], '--')
    plt.title("Final ROC Curve")
    plt.savefig('final_roc_curve.png')

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Pretrained ResNet18 structure
    model = models.resnet18(pretrained=False)
    
    # 2. MATCH THE REFINED ARCHITECTURE (Dropout + Head)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    # 3. Load refined weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    evaluate_standalone(model, val_loader, device, val_dataset.classes)
