import pytest
import torch
from model import LinearNeuralNetwork

@pytest.fixture
def model():
    return LinearNeuralNetwork()

def test_forward_pass_1(model):
    input_data = torch.randn(1, 96)
    output = model.forward(input_data)
    assert output.shape == (1, 1)

def test_forward_pass_2(model):
    input_data = torch.randn(10, 96)
    output = model.forward(input_data)
    assert output.shape == (10, 1)

def test_forward_pass_3(model):
    input_data = torch.randn(5, 96)
    output = model.forward(input_data)
    assert output.shape == (5, 1)

if __name__ == "__main__":
    pytest.main([__file__])