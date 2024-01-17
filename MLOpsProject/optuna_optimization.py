"""Optimization study for a PyTorch CNN with Optuna.

Hyperparameter optimization example of a PyTorch Convolutional Neural Network
for the MNIST dataset of handwritten digits using the hyperparameter
optimization framework Optuna.

The MNIST dataset contains 60,000 training images and 10,000 testing images,
where each sample is a small, square, 28×28 pixel grayscale image of
handwritten single digits between 0 and 9.

This script requires installing the following packages:
  torch, pandas, optuna

Author: elena-ecn
Date: 2021
"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from models.model import NeuralNetwork
from data.clean_data import *


# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, trial,dropout):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(104, 256)
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.zeros_(self.fc1.bias)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        #self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        #self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        #self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        #x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        #x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        #x = self.bn3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x



def train(network, optimizer):
    """Trains the model.

    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    for batch_i, (data, target) in enumerate(train_loader):  # For each batch

        # Limit training data for faster computation
        if batch_i * batch_size_train > number_of_train_examples:
            break

        optimizer.zero_grad()                                 # Clear gradients
        output = network(data.to(device))                     # Forward propagation
        loss = F.nll_loss(output, target.to(device))          # Compute loss (negative log likelihood: −log(y))
        loss.backward()                                       # Compute gradients
        optimizer.step()                                      # Update weights                         # Update weights


def test(network):
    """Tests the model.

    Parameters:
        - network (__main__.Net): The CNN

    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, (data, target) in enumerate(test_loader):  # For each batch

            # Limit testing data for faster computation
            if batch_i * batch_size_test > number_of_test_examples:
                break

            output = network(data.to(device))               # Forward propagation
            pred = output.data.max(1, keepdim=True)[1]      # Find max value in each row, return indexes of max values
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()  # Compute correct predictions

    accuracy_test = correct / len(test_loader.dataset)

    return accuracy_test


def objective(trial):
    """Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    dropout = trial.suggest_float("dropout", 0.2, 0.5)     # Dropout for convolutional layer 2

    # Generate the model
    model = NeuralNetwork(trial,dropout).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer)  # Train the model
        accuracy = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def data_loader(train_path: str,
                test_path: str,
                val_path: str,
                ):
    print("Loading data from: ", train_path)  
    train_loader = torch.load(train_path)
    test_loader = torch.load(test_path)
    val_loader = torch.load(val_path)
    return train_loader, test_loader, val_loader


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 10                         # Number of training epochs
    batch_size_train = 64                 # Batch size for training data
    batch_size_test = 1000                # Batch size for testing data
    number_of_trials = 100                # Number of Optuna trials
    limit_obs = True                      # Limit number of observations for faster computation

    # *** Note: For more accurate results, do not limit the observations.
    #           If not limited, however, it might take a very long time to run.
    #           Another option is to limit the number of epochs. ***

    if limit_obs:  # Limit number of observations
        number_of_train_examples = 500 * batch_size_train  # Max train observations
        number_of_test_examples = 5 * batch_size_test      # Max test observations
    else:
        number_of_train_examples = 82626                   # Max train observations
        number_of_test_examples = 12241                    # Max test observations
    # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)

    # Create directory 'files', if it doesn't exist, to save the dataset
    directory_name = 'files'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    train_path = "MLOpsProject/data/processed/train_loader.pth"
    test_path = "MLOpsProject/data/processed/test_loader.pth"
    val_path = "MLOpsProject/data/processed/val_loader.pth"
    #train_path = os.path.join(get_original_cwd(), train_path)
    #test_path = os.path.join(get_original_cwd(), test_path)
    #val_path = os.path.join(get_original_cwd(), val_path)
    #Load the data:
    train_loader,test_loader,_ = data_loader(train_path=train_path, test_path=test_path, val_path=val_path)


    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))