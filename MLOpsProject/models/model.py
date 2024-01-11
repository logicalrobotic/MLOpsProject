import torch
import torch.nn as nn

# Define the neural network class
class LinearNeuralNetwork(nn.Module):
    def __init__(self):
        super(LinearNeuralNetwork, self).__init__()
        # Define the layers
        self.linear1 = nn.Linear(in_features=96, out_features=64)  # First hidden layer
        self.linear2 = nn.Linear(in_features=64, out_features=32)  # Second hidden layer
        self.linear3 = nn.Linear(in_features=32, out_features=16)  # Third hidden layer
        self.linear4 = nn.Linear(in_features=16, out_features=8)  # Fourth hidden layer
        self.linear5 = nn.Linear(in_features=8, out_features=4)  # Fifth hidden layer
        self.output = nn.Linear(in_features=4, out_features=2)#Should we two outputs dim    # Output layer

    def forward(self, x):
        # Forward pass through the layers
        assert x.shape[1] == 96, "Input tensor should have shape (batch_size, 96)"
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = self.output(x)
        return torch.softmax(x, dim=1)

if __name__ == "__main__":
    #Create an instance of the network
    net = LinearNeuralNetwork()

    # Display the network structure
    print(net)

    #Dummy test of forward pass
    input_data = torch.randn(1, 96)
    output = net.forward(input_data)
    print("Output shape: {}".format(output.shape))
