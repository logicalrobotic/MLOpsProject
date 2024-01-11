import torch
import torch.nn as nn
from models.model import LinearNeuralNetwork
from os.path import dirname as up
from config import CSGOConfig
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import wandb
import omegaconf
import warnings
import torch
from os.path import dirname as up

# Suppress all warnings due to Hydra warnings
warnings.filterwarnings("ignore")

#Creating a config store for hydra, using the CSGOConfig class
cs = ConfigStore.instance()
cs.store(name="csgo_config", node=CSGOConfig)

#Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading network
net = LinearNeuralNetwork().to(device)

#File path to the data
one_up = up(up(__file__))
# replace '\\' with '/' for Windows
one_up = one_up.replace('\\', '/')
# Join the paths to the csv files:
file_path = one_up + "./data/processed/train_loader.pth"

def data_loader(file_path: str) -> None:
        print("Loading data from: ", file_path)
        train_loader = torch.load(file_path)
        return train_loader

#Hydra decrator to read config file
@hydra.main(config_path="conf", config_name="config")
def train(cfg: CSGOConfig) -> None:
    """Main training routine.

    Modes: debug, logging, normal
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
    
    model = net.to(device)
    
    if cfg.params.log_mode:
        wandb.watch(model,log_freq=1000)
   
    #Loading data
    train_loader = data_loader(file_path=file_path)

    #Instantiate optimizer from config file
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    
    #Loss function
    criterion = nn.CrossEntropyLoss()

    """Train loop of model with the given hyperparameters."""

    print(f"Training with learning rate {cfg.params.lr} and {cfg.params.epochs} epochs")
    for epoch in range(cfg.params.epochs):
        running_loss = 0.0   
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if cfg.params.debug_mode:
                print("outputs: ",outputs,outputs.shape, outputs.dtype)
                print("labels: ",labels.long(),labels.long().shape, labels.long().dtype)
                #print("shapes: ",outputs.shape, labels.shape, labels.unsqueeze(1).shape)
                #print("type: ",outputs.dtype,labels.dtype, labels.unsqueeze(1).dtype)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if cfg.params.log_mode:
                wandb.log({"loss": running_loss / len(train_loader)})
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    print('Finished Training')
    torch.save(model, f"trained_model.pt")
    print("Model saved at: ", f"trained_model.pt")  

if __name__ == "__main__":
    train() 