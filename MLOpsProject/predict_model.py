import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import NeuralNetwork
from data.dummy_data_loader import dummy_data
import numpy as np
from data.cs_data_loader import cs_data
from os.path import dirname as up
from data.clean_data import *
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralNetwork().to(device)

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    total = 0
    correct = 0
    model = torch.load(model)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            #torch max of outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100*(correct / total)
    print(f'Test Accuracy: {test_accuracy:.3f}%')
    return 


if __name__ == "__main__":
    one_up = up(up(__file__))
    # replace '\\' with '/' for Windows
    one_up = one_up.replace('\\', '/')
    test_path = one_up + "/data/processed/test_loader.pth"
    print("Loading data from: ", test_path)
    test_loader = torch.load(test_path)
    predict("trained_model.pt",test_loader)