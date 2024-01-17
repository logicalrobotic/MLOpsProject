
import pytest
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.preprocessing import power_transform
sys.path.append('./MLOpsProject/models')
from model import NeuralNetwork


class TestModel():
    def test_net(self):
        net = NeuralNetwork()
        input_data = torch.randn(1, 104)
        output = net.forward(input_data)
        assert output.shape == torch.Size([1, 2])

if __name__ == '__main__':
    pytest.main()