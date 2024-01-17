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


import click

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralNetwork().to(device)


# Using click to create a command line interface
@click.command()
@click.option('--model', default="models/trained_model.pt", help='Path to model')
@click.option('--dataloader', default=None, help='Path to dataloader')
@click.option('--verbose', default=False, help='Prints accuracy')
def predict(model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    verbose: bool = False):

    if model is None:
        model = up(__file__) + cfg.model_save_path
    if dataloader is None:
        dataloader = up(__file__) + cfg.test_path

    total = 0
    correct = 0
    all_predictions = []
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
            all_predictions.append(outputs)
    test_accuracy = 100*(correct / total)
    if verbose:
        print(f'Test Accuracy: {test_accuracy:.3f}%')
    return all_predictions


if __name__ == "__main__":
    predict()