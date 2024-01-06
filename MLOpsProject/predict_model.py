import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import LinearNeuralNetwork
from data.dummy_data_loader import dummy_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LinearNeuralNetwork().to(device)

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    total = 0
    correct = 0
    model = torch.load(model)
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')
    return 

if __name__ == "__main__":
    _,_,_,_,_, test_loader = dummy_data()
    predict("models/trained_model.pt",test_loader)