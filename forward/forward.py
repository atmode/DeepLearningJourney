import torch
import torch.nn as nn

class MultiLayerNN(nn.Module):
    def __init__(self):
        super(MultiLayerNN, self).__init__()
        # Define the first hidden layer (3 inputs, 2 neurons)
        self.hidden1 = nn.Linear(3, 2)
        # Define the second hidden layer (2 inputs, 3 neurons)
        self.hidden2 = nn.Linear(2, 3)
        # Define the third hidden layer (3 inputs, 2 neurons)
        self.hidden3 = nn.Linear(3, 2)
        # Define the output layer (2 inputs, 2 neurons)
        self.output = nn.Linear(2, 2)

    def forward(self, x):
        hidden1_out = torch.sigmoid(self.hidden1(x))
        hidden2_out = torch.sigmoid(self.hidden2(hidden1_out))
        hidden3_out = torch.sigmoid(self.hidden3(hidden2_out))
        output = self.output(hidden3_out)
        return hidden1_out, hidden2_out, hidden3_out, output 

model = MultiLayerNN()

with torch.no_grad():
    model.hidden1.weight = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))  # Shape: (2, 3)
    model.hidden1.bias = nn.Parameter(torch.tensor([0.1, 0.1]))  

    model.hidden2.weight = nn.Parameter(torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]))  # Shape: (3, 2)
    model.hidden2.bias = nn.Parameter(torch.tensor([0.2, 0.2, 0.2])) 

    model.hidden3.weight = nn.Parameter(torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]))  # Shape: (2, 3)
    model.hidden3.bias = nn.Parameter(torch.tensor([0.3, 0.3]))  

    model.output.weight = nn.Parameter(torch.tensor([[0.9, 1.0], [1.1, 1.2]]))  # Shape: (2, 2)
    model.output.bias = nn.Parameter(torch.tensor([0.4, 0.4]))

# Define the input
x = torch.tensor([1.0, 2.0, 3.0]) 

# Perform forward propagation
hidden1_out, hidden2_out, hidden3_out, output = model(x)

print("Output of Hidden Layer 1:", hidden1_out)
print("Output of Hidden Layer 2:", hidden2_out)
print("Output of Hidden Layer 3:", hidden3_out)
print("Final Output of the Neural Network:", output)