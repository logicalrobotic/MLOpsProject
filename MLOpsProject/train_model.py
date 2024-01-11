import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import LinearNeuralNetwork
from data.dummy_data_loader import dummy_data
from data.cs_data_loader import cs_data
from os.path import dirname as up
from config import CSGOConfig
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import wandb
import omegaconf

#Creating a config store for hydra, using the CSGOConfig class
cs = ConfigStore.instance()
cs.store(name="csgo_config", node=CSGOConfig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = LinearNeuralNetwork().to(device)



#Hyperparameters
#lr = 0.001
#batch_size = 10
#epochs = 5
#model_save_path = f"models/trained_model.pt"

# Loss function and optimizer
#criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
#optimizer = optim.Adam(net.parameters(), lr=lr)


#Hydra decrator to read config file
@hydra.main(config_path="conf", config_name="config")
def train(cfg: CSGOConfig) -> None:
    #print(cfg.params)
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.config.update({"lr": cfg.params.lr, "epochs": cfg.params.epochs})

    wandb.init(project="csgo")
    
    model = net.to(device)
    wandb.watch(model,log_freq=1000)
    #print(cfg.optimizer)

    """This is getting removed in future versions:start"""
    #optimizer = cfg.optim.optimizer(model.parameters(), lr=cfg.params.lr)
    #criterion = cfg.loss.criterion()
    #optimizer = optim.Adam(model.parameters(), lr=cfg.params.lr)
    criterion = nn.CrossEntropyLoss()
    #return
    one_up = up(up(__file__))
    # replace '\\' with '/' for Windows
    one_up = one_up.replace('\\', '/')
    # Join the paths to the csv files:
    file_path = one_up + "/data/processed/train_loader.pth"
    print("Loading data from: ", file_path)
    train_loader = torch.load(file_path)
    
    #train_loader, _, _ = cs_data(file_path=file_path)

    #_,_,_,_,train_loader, _ = dummy_data(batch_size=cfg.params.batch_size)

    """This is getting removed in future versions:end"""
    #optimizer = cfg.optimizer._target_(model.parameters(), lr=cfg.params.lr)
    #print(instantiate(cfg.optimizer, params=model.parameters()))
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    """Train a model."""
    print(f"Training with learning rate {cfg.params.lr} and {cfg.params.epochs} epochs")
    for epoch in range(cfg.params.epochs):
        running_loss = 0.0   
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("outputs: ",outputs,outputs.shape, outputs.dtype)
            print("labels: ",labels.long(),labels.long().shape, labels.long().dtype)
                #print("shapes: ",outputs.shape, labels.shape, labels.unsqueeze(1).shape)
                #print("type: ",outputs.dtype,labels.dtype, labels.unsqueeze(1).dtype)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"loss": running_loss / len(train_loader)})
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    print('Finished Training')
    torch.save(model, f"trained_model.pt")
    print("Model saved at: ", f"trained_model.pt")  

if __name__ == "__main__":
    #_,_,_,_,train_loader, _ = dummy_data(batch_size=batch_size)
    #train(net, train_loader, criterion, optimizer)
    #train()
    


    train() 