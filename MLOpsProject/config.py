from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim


#@dataclass
#class Paths:
    #log: str
    #data: str


#@dataclass
#class Files:
   # train_data: str
    #train_labels: str
    #test_data: str
    #test_labels: str


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int

@dataclass
class Optim:
    optim : torch.optim.Optimizer

@dataclass
class Loss:
    loss : torch.nn.Module


#@dataclass
#class MNISTConfig:
    #paths: Paths
    #files: Files
    #params: Params