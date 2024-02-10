import numpy as np

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
    
   