import pandas as pd
import torch

from os.path import dirname as up
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Create a custom dataset for the data where first 96 columns are the features and the last column is the label:
class CS_Dataset(Dataset):
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.iloc[:, 1:-1]
        self.y = self.data.iloc[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values).type(torch.FloatTensor), torch.tensor(self.y.iloc[idx]).type(torch.FloatTensor)

# Split the data into train, validation and test sets:
def cs_data(file_path: str, batch_size: int=64, split: list=[0.6, 0.2, 0.2], shuffle: list=[True, False, False], state: int=42) -> None:
    dataset = CS_Dataset(file_path)
    train, test = train_test_split(dataset, test_size=split[1], random_state=state)
    train, val = train_test_split(train, test_size=split[1], random_state=state)

    # Create Train, Validation and Test dataloaders:
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle[0])
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle[1])
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle[2])
    torch.save(train_loader, 'MLOpsProject/data/processed/train_loader.pth')
    torch.save(val_loader, 'MLOpsProject/data/processed/val_loader.pth')
    torch.save(test_loader, 'MLOpsProject/data/processed/test_loader.pth')
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Get the current directory with two levels up:
    two_up = up(up(up(__file__)))
    # replace '\\' with '/' for Windows
    two_up = two_up.replace('\\', '/')
    # Join the paths to the csv files:
    file_path = two_up + "/data/processed/csgo_converted.csv"

    # Create the dataloaders:
    train_loader, val_loader, test_loader = cs_data(file_path)

    # Print the size of the dataloaders:
    print("Train loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))