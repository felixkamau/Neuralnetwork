'''
let's write a simple test to verify that our neural network is able to learn 
the XOR function. The XOR function takes two binary inputs (0 or 1) and returns 1 
if the inputs are different and 0 if they are the same.

'''
'''
This test will train the neural network using the XOR dataset and then evaluate 
its performance on each input example. It prints out the input, target, and predicted 
output for each example, allowing you to see how well the neural network is able to learn 
the XOR function

'''

# Import the NeuralNetwork class
from neural import NeuralNetwork
import numpy as np

# Create an instance of the NeuralNetwork class
input_size = 2
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Prepare inputs and corresponding targets for the XOR function
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR dataset
targets = np.array([[0], [1], [1], [0]]) # actual output of logic gate

# Train the neural network
epochs = 10000
learning_rate = 0.1
nn.train(inputs, targets, epochs, learning_rate)

# After training, test the neural network
for i in range(len(inputs)):
    input_example = inputs[i]
    target_example = targets[i]
    output_example = nn.forward_propagation(input_example)
    print(f'Input: {input_example}, Target: {target_example}, Predicted: {output_example[0]}')


