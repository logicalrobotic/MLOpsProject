import torch
import torch.nn as nn
from models.model import NeuralNetwork
from os.path import dirname as up
from config import CSGOConfig
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import wandb
import omegaconf
import warnings
import torch
from sklearn.model_selection import train_test_split
from os.path import dirname as up
from data.clean_data import *
from torchvision import datasets, transforms

# Suppress all warnings due to Hydra warnings
warnings.filterwarnings("ignore")

#Creating a config store for hydra, using the CSGOConfig class
cs = ConfigStore.instance()
cs.store(name="csgo_config", node=CSGOConfig)

#Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading network
net = NeuralNetwork()

#File path to the data
one_up = up(up(__file__))
# replace '\\' with '/' for Windows
one_up = one_up.replace('\\', '/')
# Join the paths to the csv files:
train_path = one_up + "/data/processed/train_loader.pth"
test_path = one_up + "/data/processed/test_loader.pth"
val_path = one_up + "/data/processed/val_loader.pth"

#Normalize data and return as tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def data_loader(train_path: str,
                test_path: str,
                val_path: str,
                ) -> None:
    print("Loading data from: ", train_path)
    train_loader = torch.load(train_path)
    test_loader = torch.load(test_path)
    val_loader = torch.load(val_path)
    return train_loader, test_loader, val_loader

#Hydra decrator to read config file
@hydra.main(config_path="conf", config_name="config")
def train(cfg: CSGOConfig) -> None:
    """Main training routine.

    Modes: debug, logging, normal. Change mode in config file.
        -> debug: prints out the shapes of the input and output tensors
        -> logging: logs the loss and accuracy to wandb
        -> normal: no logging or debugging. Defaults to this mode if no mode is specified.

    Args: see config file (conf/config.yaml))

    Returns: None

    """

    #Initiate model + wandb logging if log_mode is true
    if cfg.params.log_mode:
        wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
        wandb.config.update({"lr": cfg.params.lr, "epochs": cfg.params.epochs})

    if cfg.params.log_mode:
        wandb.init(project="csgo")
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    
    if cfg.params.log_mode:
        wandb.watch(model,log_freq=1000)
   
    #Loading data
    train_loader,_,val_loader = data_loader(train_path=train_path, test_path=test_path, val_path=val_path)
    
    #Instantiate optimizer from config file
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    
    #Loss function
    criterion = nn.CrossEntropyLoss()

    """Train loop of model with the given hyperparameters."""

    print(f"Training with learning rate {cfg.params.lr} and {cfg.params.epochs} epochs")
    for epoch in range(cfg.params.epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if cfg.params.debug_mode:
                print("outputs: ",outputs,outputs.shape, outputs.dtype)
                print("labels: ",labels.long(),labels.long().shape, labels.long().dtype)
                #print("shapes: ",outputs.shape, labels.shape, labels.unsqueeze(1).shape)
                #print("type: ",outputs.dtype,labels.dtype, labels.unsqueeze(1).dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if cfg.params.log_mode:
                wandb.log({"loss": running_loss / len(train_loader)})
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.long().to(device)
                outputs = model(images.float())  # Convert images to floats
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch+1}/{cfg.params.epochs}, Validation Accuracy: {accuracy:.4f}')
        if cfg.params.log_mode:
            wandb.log({"accuracy": accuracy})
    print('Finished Training')
    torch.save(model, f"trained_model.pt")
    print("Model saved at: ", f"trained_model.pt")  

if __name__ == "__main__":
    train() 