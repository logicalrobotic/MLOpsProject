import torch
from data.clean_data import *
from helper import *

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    verbose: bool = False,
    logging: bool = False,
) -> list:
    
    """Main prediction routine.

    Modes: normal, verbose,logging. Change mode in function input.
        -> normal: no logging or debugging. Defaults to this mode if no mode is specified.
        -> verbose: prints the device used and the test accuracy
        -> logging: logs the loss and accuracy to wandb

    Returns: List of softmax of the model on the test set

    """

    #Move the newest model to the models folder
    source_folder = "./outputs/"
    destination_folder = "./MLOpsProject/models"
    newest_model = find_newest_pt_model(source_folder)
    move_model_to_folder(newest_model, destination_folder,verbose = verbose)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device: ", device)
    
    """Prediction loop of model"""

    total = 0
    correct = 0
    all_predictions = []
    model = torch.load(model)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.append(outputs)
    test_accuracy = 100*(correct / total)
    if verbose:
        print(f'Test Accuracy: {test_accuracy:.3f}%')
    return all_predictions


if __name__ == "__main__":
    test_path = "MLOpsProject/data/processed/test_loader.pth"
    print("Loading data from: ", test_path)
    test_loader = torch.load(test_path)
    predict("MLOpsProject/models/trained_model.pt",test_loader,verbose=True)