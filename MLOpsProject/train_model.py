import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import LinearNeuralNetwork
from data.dummy_data_loader import dummy_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = LinearNeuralNetwork().to(device)

#Hyperparameters
lr = 0.001
batch_size = 10
epochs = 5

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(net.parameters(), lr=lr)



# Training loop
def train(net, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()   
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

if __name__ == "__main__":
    _,_,_,_,train_loader, _ = dummy_data(batch_size=batch_size)
    train(net, train_loader, criterion, optimizer, epochs=epochs)