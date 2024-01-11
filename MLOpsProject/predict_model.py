import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import LinearNeuralNetwork
from data.dummy_data_loader import dummy_data
import numpy as np
from data.cs_data_loader import cs_data
from os.path import dirname as up

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LinearNeuralNetwork().to(device)

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    total = 0
    correct = []
    model = torch.load(model)
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            #torch max of outputs
            _, predicted = torch.max(outputs.data, 1)
            correct.append((predicted == labels).sum().item())
            #print(correct)
    print(f'Accuracy: {np.mean(correct)}%')
    return 

if __name__ == "__main__":
    #_,_,_,_,_, test_loader = dummy_data()
    one_up = up(up(__file__))
    # replace '\\' with '/' for Windows
    one_up = one_up.replace('\\', '/')
    # Join the paths to the csv files:
    file_path = one_up + "/data/processed/train_loader.pth"
    print("Loading data from: ", file_path)
    train_loader = torch.load(file_path)
    file_path = one_up + "/data/processed/test_loader.pth"
    print("Loading data from: ", file_path)
    test_loader = torch.load(file_path)
    print("Data loaded successfully!")
    print("Predicting on train data")
    predict("trained_model.pt",train_loader)
    print("Predicting on test data")
    predict("trained_model.pt",test_loader)