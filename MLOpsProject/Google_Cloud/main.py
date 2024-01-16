import torch
import sys
sys.path.append('./models')
from model import NeuralNetwork
from os.path import dirname as up
sys.path.append('./data')
from clean_data import *
sys.path.append('./')
from flask import Flask, request, jsonify
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralNetwork().to(device)

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    total = 0
    correct = 0
    model = torch.load(model)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            #torch max of outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100*(correct / total)
    print(f'Test Accuracy: {test_accuracy:.3f}%')
    return

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        data = np.array(data)
        data = torch.from_numpy(data)
        data = data.float()
        data = data.to(device)
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        return jsonify(predicted.item())

if __name__ == "__main__":
    app.run()
#if __name__ == "__main__":
#    one_up = up(up(up(__file__)))
#    # replace '\\' with '/' for Windows
#    one_up = one_up.replace('\\', '/')
#    test_path = one_up + "/data/processed/test_loader.pth"
#    print("Loading data from: ", test_path)
#    test_loader = torch.load(test_path)
#    predict("Google_Cloud/trained_model.pt",test_loader)