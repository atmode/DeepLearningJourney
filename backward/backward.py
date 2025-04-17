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
        return output

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

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  

# Define the target output
y = torch.tensor([1.0, 0.0]) 

# Define the loss function (Mean Squared Error)
mse_loss = nn.MSELoss()

# Perform forward propagation
output = model(x)

# Compute the loss
loss = mse_loss(output, y)

# Perform backward propagation
loss.backward()

print("Gradients of hidden1 weights:", model.hidden1.weight.grad)
print("Gradients of hidden1 biases:", model.hidden1.bias.grad)

print("Gradients of hidden2 weights:", model.hidden2.weight.grad)
print("Gradients of hidden2 biases:", model.hidden2.bias.grad)

print("Gradients of hidden3 weights:", model.hidden3.weight.grad)
print("Gradients of hidden3 biases:", model.hidden3.bias.grad)

print("Gradients of output weights:", model.output.weight.grad)
print("Gradients of output biases:", model.output.bias.grad)
