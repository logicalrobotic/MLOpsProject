import torch
from torch.utils.data import DataLoader, TensorDataset    

def dummy_data(batch_size=10):
# Dummy dataset (for demonstration purposes)
    X_train = torch.randn(100, 96)  # 100 samples, 96 features each
    y_train = torch.randint(0, 2, (100, 1)).type(torch.FloatTensor)  # 100 binary labels

    X_test = torch.randn(50, 96)  # 50 samples, 96 features each
    y_test = torch.randint(0, 2, (50, 1)).type(torch.FloatTensor)  # 50 binary labels

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return X_train, y_train, X_test, y_test, train_loader, test_loader

if __name__ == "__main__":
    X_train, y_train, X_test, y_test,train_loader, test_loader = dummy_data(batch_size=10)
    #Make some nice, informative prints for the user
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"train_loader length: {len(train_loader)}")
    print(f"test_loader length: {len(test_loader)}")

    
    