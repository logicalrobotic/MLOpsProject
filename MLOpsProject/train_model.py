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
model_save_path = f"models/trained_model.pt"

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(net.parameters(), lr=lr)


# Training loop
def train(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epochs: int
        ) -> None:
    """Train a model."""
    print(f"Training with learning rate {lr} and epochs {epochs}")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0   
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    print('Finished Training')
    torch.save(model, model_save_path)

if __name__ == "__main__":
    _,_,_,_,train_loader, _ = dummy_data(batch_size=batch_size)
    train(net, train_loader, criterion, optimizer, epochs=epochs)