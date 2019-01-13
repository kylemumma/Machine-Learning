import numpy

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it genereates the same numbers
        # every time the program runs
        numpy.random.seed(1)

        # Creates a 3x1 matrix with values in the range of -1 to 1
        self.weights = 2 * numpy.random.rand(3, 1) - 1

    # sigmoid function, we pass it an x value and it normalizes it
    # we will be passing it the weighted sum of inputs to normalize
    def sigmoid_function(self, x):
        return 1 / (1 + numpy.exp(-x))

    # derivative of sigmoid function
    # we will be passing it the outputs
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # we train the network using trial and error
    # it adjusts its weights each time based on error margin
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # pass the training inputs though neural network (see what it gets for outputs)
            output = self.think(training_set_inputs)

            # calculate the error (difference between desired output and actual output) (matrix subtraction)
            error = training_set_outputs - output

            # adjusted weights = error * input * sigmoid curve gradient
            adjustment = numpy.dot(training_set_inputs.T, error * self.sigmoid_derivative(output))

            # adjust the weights
            self.weights += adjustment

    # neural network calculates outputs based on inputs
    def think(self, inputs):
        #get weighted sum of inputs
        inputs_times_weights = numpy.dot(inputs, self.weights)

        #normalizes weighted sum of inputs
        return self.sigmoid_function(inputs_times_weights)

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random starting weights: ")
    print(neural_network.weights)

    # 4 examples each with 3 inputs and 1 output
    training_set_inputs = numpy.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = numpy.array([[0, 1, 1, 0]]).T

    # train network 10,000 times
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New weights after training: ")
    print(neural_network.weights)

    print("New situation: [0, 0, 0] Output:")
    print(neural_network.think(numpy.array([1, 0, 0])))
