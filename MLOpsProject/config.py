from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class Files:
    model_save_path : str


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int

@dataclass
class Optim:
    optim : str

@dataclass
class Loss:
    loss : str


@dataclass
class CSGOConfig:
    params: Params
    optim: Optim
    loss: Loss