import numpy as np
#THE TESTING OF THE NN IN THE MAIN.py FILE


class NeuralNetwork:
    """
        NeuralNettwwork class represents a neural network with an input layer, a hidden layer, and an output layer.

        Attributes:
            weights_input_hidden (ndarray): The weights connecting the input layer to the hidden layer.
            bias_input_hidden (ndarray): The bias values for the hidden layer.
            weights_hidden_output (ndarray): The weights connecting the hidden layer to the output layer.
            bias_hidden_output (ndarray): The bias values for the output layer.

        Methods:
            __init__(self, input_size, hidden_size, output_size): Initializes the neural network with random weights and biases.

    """
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size,hidden_size)
        # Hidden size represent the number of neurons, it generates
        # an array of one dimension
        self.bias_input_hidden = np.random.randn(1,hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.random.randn(1,output_size)
        
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # we define a sigmoid function and it's  derivtive
        # The sigmoid function introduces a non-linearity into 
        # the network  allowing to learn complex patterns
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    def forward_propagation(self, inputs):
        """
            Compute the output of the neural network given the input.

            Args:
                inputs (ndarray): The input data as a 2D array of shape (n_samples, input_size).

            Returns:
                ndarray: The output of the neural network as a 2D array of shape (n_samples, output_size).

            Notes:
                - The number of samples is equal to the number of rows in the input array.
                - The number of input neurons is equal to the number of columns in the input array.
                - The number of output neurons is equal to the number of columns in the output array.
                - The number of hidden neurons is equal to the number of columns in the input array.

        """
        #  The forward  method computes the output  of the neural net
        #  given the input.
        #  The input is a 2D array of shape (n_samples, input_size)
        #  The output is a 2D array of shape (n_samples, output_size)
        #  The number of samples is equal to the number of rows in the input array
        #  The number of input neurons is equal to the number of columns in the input array
        #  The number of output neurons is equal to the number of columns in the output array
        #  The number of hidden neurons is equal to the number of columns in the input array
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        final_output = self.sigmoid(output_input)
        return final_output
    
    def backward_propagation(self,inputs, targets, outputs, learning_rate):
        """
        Calculate the gradients and update the weights and biases during the backward propagation step of the neural network.

        Args:
            inputs (ndarray): The input data as a 2D array of shape (n_samples, input_size).
            targets (ndarray): The target output data as a 2D array of shape (n_samples, output_size).
            outputs (ndarray): The predicted output data as a 2D array of shape (n_samples, output_size).
            learning_rate (float): The learning rate for updating the weights and biases.

        Returns:
            None

        Notes:
            - The number of samples is equal to the number of rows in the input, targets, and outputs arrays.
            - The number of input neurons is equal to the number of columns in the inputs array.
            - The number of output neurons is equal to the number of columns in the targets and outputs arrays.
            - The number of hidden neurons is equal to the number of columns in the inputs array.

        Algorithm:
            1. Calculate the output errors by subtracting the targets from the outputs.
            2. Calculate the output gradients by multiplying the output errors with the derivative of the sigmoid function applied to the outputs.
            3. Calculate the hidden errors by dot product of the output gradients with the transpose of the weights connecting the hidden layer to the output layer.
            4. Calculate the hidden gradients by multiplying the hidden errors with the derivative of the sigmoid function applied to the outputs.
            5. Update the weights and biases:
                - Update the weights connecting the hidden layer to the output layer by adding the dot product of the transpose of the inputs with the output gradients.
                - Update the biases for the output layer by adding the sum of the output gradients along the rows, multiplied by the learning rate.
                - Update the weights connecting the input layer to the hidden layer by adding the dot product of the hidden gradients with the transpose of the inputs, multiplied by the learning rate.
                - Update the biases for the hidden layer by adding the sum of the hidden gradients along the rows, multiplied by the learning rate.
        """
      
            # Calculate the output errors
        output_errors = targets - outputs
        
        # Calculate the output gradients
        output_gradients = output_errors * self.sigmoid_derivative(outputs)
        
        # Calculate the hidden errors
        hidden_errors = np.dot(output_gradients, self.weights_hidden_output.T)
        
        # Calculate the hidden gradients
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        hidden_output = self.sigmoid(hidden_input)
        hidden_gradients = hidden_errors * self.sigmoid_derivative(hidden_output)
        
        # Update the weights and biases
        self.weights_hidden_output += np.dot(hidden_output.T, output_gradients) * learning_rate
        self.bias_hidden_output += np.sum(output_gradients, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_gradients) * learning_rate
        self.bias_input_hidden += np.sum(hidden_gradients, axis=0, keepdims=True) * learning_rate
            
    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            outputs = self.forward_propagation(inputs)
            self.backward_propagation(inputs, targets, outputs, learning_rate)
            loss = np.mean(np.square(targets - outputs))
            print(f'Epoch: {epoch}, Loss: {loss}')