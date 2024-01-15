from dataclasses import dataclass

@dataclass
class Files:
    model_save_path : str
    train_path : str
    test_path : str
    val_path : str

@dataclass
class Params:
    epoch_count: int
    batch_size: int
    debug_mode: bool
    log_mode: bool

@dataclass
class Optim:
    optim : str
    lr: float

@dataclass
class Loss:
    loss : str

@dataclass
class CSGOConfig:
    params: Params
    optim: Optim
    loss: Loss