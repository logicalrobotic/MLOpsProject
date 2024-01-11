import pandas as pd
from os.path import dirname as up
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import OneHotEncoder, power_transform
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def remove_grenades(df):
    """
    Removes grenade columns from the dataframe
    """
    # Get the columns with grenade info
    cols_grenade = df.columns[df.columns.str.contains('grenade')]

    # Drop the columns
    df = df.drop(cols_grenade, axis=1)

    return df

def encode_targets(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    return y_encoded

def encode_inputs(X, object_cols):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_encoded = pd.DataFrame(ohe.fit_transform(X[object_cols]))
    X_encoded.columns = ohe.get_feature_names_out(object_cols)
    X_encoded.index = X.index
    return X_encoded

def yeo_johnson(series):
    arr = np.array(series).reshape(-1, 1)
    return power_transform(arr, method='yeo-johnson')

if __name__ == "__main__":
    # Load the data

    # Get the data and process it
    two_up = up(up(up(__file__)))
    # replace '\\' with '/' for Windows
    two_up = two_up.replace('\\', '/')

    # Join the paths to the csv files:
    filename = two_up + "/data/raw/csgo_round_snapshots.csv"
    output_filename = two_up + "/data/processed/csgo_converted.csv"

    print("Reading csv file from: ", filename)

    # Read the csv file
    df = pd.read_csv(filename)

    # Split X and y
    y = df.round_winner
    X = df.drop(['round_winner'], axis=1)

    # Drop columns with grenade info
    #X = remove_grenades(X)

    print(f"Total number of samples: {len(X)}")
    print(X.head())

    # Use OH encoder to encode predictors
    object_cols = ['map', 'bomb_planted']
    X_encoded = encode_inputs(X, object_cols)
    numerical_X = X.drop(object_cols, axis=1)
    X = pd.concat([numerical_X, X_encoded], axis=1)

    # Use label encoder to encode targets
    y = encode_targets(y)

    # Make data more Gaussian-like
    cols = ['time_left', 'ct_money', 't_money', 'ct_health',
    't_health', 'ct_armor', 't_armor', 'ct_helmets', 't_helmets',
    'ct_defuse_kits', 'ct_players_alive', 't_players_alive']
    for col in cols:
        X[col] = yeo_johnson(X[col])

    print(f"Total number of samples: {len(X)}")
    print(X.head())

    # Make a train, validation and test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
                                                            stratify=y, test_size=0.1, 
                                                            random_state=0)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,
                                                            stratify=y_train_full, 
                                                            test_size=0.25, random_state=0)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create the train, validation, and test datasets
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    valid_dataset = CustomDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

    # Create the train, validation, and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Save the dataloaders
    torch.save(train_dataloader, two_up + "/data/processed/train_loader.pth")
    torch.save(valid_dataloader, two_up + "/data/processed/val_loader.pth")
    torch.save(test_dataloader, two_up + "/data/processed/test_loader.pth")

    # Print the number of samples in each dataset
    print(f"Number of samples in train dataset: {len(train_dataset)}")
    print(f"Number of samples in validation dataset: {len(valid_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")