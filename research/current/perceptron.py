import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activate(summation)

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights[0] += error * self.lr
        self.weights[1:] += error * inputs * self.lr #does perceptron have synapses?

        return error 

#what can we teach to a perceptron?
    
    

    