import torch
import torch.nn as nn

# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(104, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    #Create an instance of the network
    net = NeuralNetwork()

    # Display the network structure
    print(net)

    #Dummy test of forward pass
    input_data = torch.randn(1, 104)
    output = net.forward(input_data)
    print("Output shape: {}".format(output.shape))
