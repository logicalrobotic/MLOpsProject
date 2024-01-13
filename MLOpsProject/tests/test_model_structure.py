import pytest
import torch
import sys
sys.path.append('./models')
from model import NeuralNetwork

@pytest.fixture
def model():
    return NeuralNetwork()

def test_forward_pass_1(model):
    input_data = torch.randn(1, 104)
    output = model.forward(input_data)
    assert output.shape == (1, 2)

def test_forward_pass_2(model):
    input_data = torch.randn(10, 104)
    output = model.forward(input_data)
    assert output.shape == (10, 2)

def test_forward_pass_3(model):
    input_data = torch.randn(5, 104)
    output = model.forward(input_data)
    assert output.shape == (5, 2)

if __name__ == "__main__":
    pytest.main([__file__])