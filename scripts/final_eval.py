import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from scripts.train_cnn import evaluate_model, NUM_CLASSES, DATA_DIR, BATCH_SIZE

# Data Transforms (same as training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model structure
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load weights
    MODEL_PATH = "dental_cnn_model.pth"
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    
    # Load validation data
    val_dir = os.path.join(DATA_DIR, 'val')
    val_dataset = datasets.ImageFolder(val_dir, data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define global class_names needed by evaluate_model
    import scripts.train_cnn
    scripts.train_cnn.class_names = val_dataset.classes
    scripts.train_cnn.device = device
    
    print("Running Final Evaluation...")
    evaluate_model(model, val_loader)
