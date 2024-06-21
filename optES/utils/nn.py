
from collections import OrderedDict 
import torch
import random
import numpy as np

class NeuralNet(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dim_layers, activation="relu"):
        """
        Initializes a neural network with `dim_input` inputs, `dim_output` outputs, and `dim_layers` hidden layers.

        Args:
            dim_input (int): Number of inputs to the network.
            dim_output (int): Number of outputs from the network.
            dim_layers (list): List of integers representing the number of neurons in each hidden layer.
            activation_func (string): Activation function to use in the hidden layers ["relu", "sigmoid", "tanh", "softplus", "elu", "relu6", "leakyrelu"].
        """
        super(NeuralNet, self).__init__()

        # Activation function
        if activation == "relu":
            activation_func = torch.nn.ReLU
        elif activation == "sigmoid":
            activation_func = torch.nn.Sigmoid
        elif activation == "tanh":
            activation_func = torch.nn.Tanh 
        elif activation == "softplus":
            activation_func = torch.nn.Softplus 
        elif activation == "elu":
            activation_func = torch.nn.ELU  
        elif activation == "relu6":
            activation_func = torch.nn.ReLU6
        elif activation == "leakyrelu":
            activation_func = torch.nn.LeakyReLU
 
        list_of_layers = []

        if len(dim_layers) == 0:
            dim_layers = [dim_output] 

        # append input layer
        list_of_layers.append(("input_layer", torch.nn.Linear(dim_input, dim_layers[0])))
        list_of_layers.append(("input_activation", activation_func()))

        # append hidden layers
        if len(dim_layers) > 1:
            for i in range(len(dim_layers)-1):
                list_of_layers.append((f"linear_layer_{i}", torch.nn.Linear(dim_layers[i], dim_layers[i+1])))
                list_of_layers.append((f"activation_layer_{i}", activation_func()))

        # append input layer
        list_of_layers.append(("output_layer", torch.nn.Linear(dim_layers[-1], dim_output)))
        list_of_layers.append(("output_activation", torch.nn.ReLU()))

        self.layers = torch.nn.Sequential(OrderedDict(list_of_layers))

        print(self.layers)

    def forward(self, x):
        return self.layers(x)