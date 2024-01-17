import pytest
import torch
import sys
sys.path.append('./MLOpsProject/models')
from model import NeuralNetwork

class TestModelStructure:
    def setup_method(self):
        # Instantiate the model before each test
        self.model = NeuralNetwork()

    def test_forward_pass_1(self):
        input_data = torch.randn(1, 104)
        output = self.model.forward(input_data)
        assert output.shape == (1, 2)

    def test_forward_pass_2(self):
        input_data = torch.randn(10, 104)
        output = self.model.forward(input_data)
        assert output.shape == (10, 2)

    def test_forward_pass_3(self):
        input_data = torch.randn(5, 104)
        output = self.model.forward(input_data)
        assert output.shape == (5, 2)

if __name__ == "__main__":
    pytest.main()