"""
    2802ICT Intelligent Systems, Assignment #2, Neural Network for MNIST Digits.
    Written by Jack Kearsley (s5007230), May 2017.

    Please place all of the csv.gz files in the csv-input folder.
"""

import numpy as np
import sys

# Ignores float overflow errors from numpy when dealing with exp()
np.seterr(over='ignore')

class Neural_Net(object):

    def __init__(self):
        # Gets the command-line arguments, CSV files are stored in the 'csv-input' directory.
        if len(sys.argv) == 8:
            self.N_Input = int(sys.argv[1])
            self.N_Hidden = int(sys.argv[2])
            self.N_Output = int(sys.argv[3])
            self.Train_Set = "csv-input/" + sys.argv[4]
            self.Train_Set_Label = "csv-input/" + sys.argv[5]
            self.Test_Set = "csv-input/" + sys.argv[6]
            self.Test_Set_Prediction = "csv-output/" + sys.argv[7]

        else:
            print("Not enough arguments. Usage: neural_network.py NInput NHidden NOutput TrainDigitX.csv.gz" +
                  "TrainDigitY.csv.gz TestDigitX.csv.gz PredictDigitY.csv.gz")
            sys.exit()

        # N_Input must be 784 and N_Output must be 10 to work with MNIST data sets.
        if self.N_Input != 784 or self.N_Output != 10:
            print("Warning: For use with MNIST digits, N_Input parameter should be 784, and N_Output should be 10.")

        # Reads in TrainDigitX.csv.gz
        print("Reading in '" + self.Train_Set + "', please wait about 1 minute...")
        self.Train_X = np.loadtxt(self.Train_Set, delimiter=',')

        # Reads in TrainDigitY.csv.gz
        print("Reading in '" + self.Train_Set_Label + "'...")
        self.Raw_Labels = np.loadtxt(self.Train_Set_Label)
        self.num_samples = self.Raw_Labels.shape[0]
        print("Number of samples: %d" % self.num_samples)

        # Converts TrainDigitY to an array with 'num_samples' rows and 10 columns (for each of the output neurons).
        # The index of the value will be given a '1'.
        self.Train_Y = np.zeros((self.num_samples, self.N_Output))
        for i in range(self.num_samples):
            value = int(self.Raw_Labels[i])
            self.Train_Y[i][value] = 1

        # Loads in TestSetX.csv.gz.
        self.Test_X = np.loadtxt(self.Test_Set, delimiter=',')

        # The synapses in layer 0 (between input and hidden layer).
        # Values between [-1..1] with mean of 0.
        # A matrix 784 rows by 30 cols.
        self.synapses0 = 2*np.random.random((self.N_Input, self.N_Hidden))-1

        # The synapses in layer 1 (between hidden and output layer).
        # Values between [-1..1] with mean of 0.
        # A matrix 30 rows by 10 cols.
        self.synapses1 = 2*np.random.random((self.N_Hidden, self.N_Output))-1

        # Bias weights. Values between [-1..1] with mean of 0.
        self.bias0 = 2 * np.random.random((1, self.N_Hidden)) -1
        self.bias1 = 2 * np.random.random((1, self.N_Output)) -1

        np.random.seed(1)

    # Trains the neural network with back propagation and stochastic gradient descent.
    def train(self, epochs, mini_batch_size, learning_rate, bias_input):
        print("Training neural network...")

        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.bias_input = bias_input

        for e in range(self.epochs):

            # Creates a matrix of random indices, these indices are then used from Train_X as the mini_batch samples.
            # The random selection uses no replacement, so each sample is only used once.
            mb_indexes = np.random.choice(self.num_samples, size=((self.num_samples // self.mini_batch_size), self.mini_batch_size), replace=False)

            # For each mini-batch in the training set.
            for m in range(self.num_samples//self.mini_batch_size):

                mini_batch = self.Train_X[mb_indexes[m]]
                mini_batch_label = self.Train_Y[mb_indexes[m]]

                # Forward propagation.
                X = mini_batch
                Y = mini_batch_label

                z2 = np.dot(X, self.synapses0)
                z2 += self.bias_input * self.bias0
                a2 = self.nonlin(z2)

                z3 = np.dot(a2, self.synapses1)
                z3 += self.bias_input * self.bias1
                y_hat = self.nonlin(z3)

                # Back Propagation - Quadratic cost method.
                l2_error = -1 * (Y - y_hat)
                delta3 = np.multiply(l2_error, self.nonlin(z3, deriv=True))
                l1_error = delta3.dot(self.synapses1.T)
                delta2 = np.multiply(l1_error, self.nonlin(z2, deriv=True))

                dEdW2 = np.dot(a2.T, delta3)
                dEdW1 = np.dot(X.T, delta2)
                dEdB2 = np.dot(np.zeros((1, mini_batch.shape[0])) + self.bias_input, delta3)
                dEdB1 = np.dot(np.zeros((1, mini_batch.shape[0])) + self.bias_input, delta2)

                """
                # Back Propagation - Cross-entropy method.
                dEdW2 = np.dot(a2.T, (y_hat - Y))

                A = np.dot((y_hat - Y), self.synapses1.T) * (a2 * (1 - a2))
                D = np.dot(A.T, X)
                dEdW1 = D.T

                dEdB2 = np.dot(np.zeros((1, mini_batch.shape[0])) + self.bias_input, (y_hat - Y))
                dEdB1 = np.dot(np.zeros((1, mini_batch.shape[0])) + self.bias_input, A)
                """

                # Update the weights in each layer.
                self.synapses1 -= (self.learning_rate/self.mini_batch_size) * dEdW2
                self.synapses0 -= (self.learning_rate/self.mini_batch_size) * dEdW1

                # Update the bias weights.
                self.bias1 -= (self.learning_rate/self.mini_batch_size) * dEdB2
                self.bias0 -= (self.learning_rate/self.mini_batch_size) * dEdB1

                """
                # At every 10% interval, prints out the error value.
                if (e % (self.epochs / 10)) == 0 and m == 0:
                    percent_done = (e / self.epochs) * 100
                    print("%.2f %% done" % percent_done)
                    print("Error: " + str(0.5 * np.mean(np.abs(l2_error) ** 2)))
                    print("Quad cost: " + str(self.quad_cost(mini_batch_label, y_hat)))
                    print("Cross-entropy cost: " + str(self.cross_ent_cost(mini_batch_label, y_hat)))
                """

        print("Training finished.")


    # Resets the weights to random values. Used for testing parameters iteratively.
    def reset_weights(self):
        self.synapses0 = 2 * np.random.random((self.N_Input, self.N_Hidden)) - 1
        self.synapses1 = 2 * np.random.random((self.N_Hidden, self.N_Output)) - 1
        self.bias0 = 2 * np.random.random((1, self.N_Hidden)) - 1
        self.bias1 = 2 * np.random.random((1, self.N_Output)) - 1

    # Sigmoid function. When deriv=True, the sigmoid-derivative function is applied.
    def nonlin(self, x, deriv=False):
        if (deriv==True):
            return self.nonlin(x) * (1 - self.nonlin(x))
        return 1.0/(1.0+np.exp(-x))

    # Forward propagates an input matrix (X) through the neural network.
    def fwd_prop(self, X):
        l0 = X
        l1 = self.nonlin(np.dot(l0, self.synapses0))
        l2 = self.nonlin(np.dot(l1, self.synapses1))
        return l2

    # Quadratic cost function.
    def quad_cost(self, y, y_hat):
        return (1/(2*self.num_samples)) * sum((np.sum((np.abs(y - y_hat)), axis=0)) ** 2)

    # Cross-entropy cost function.
    def cross_ent_cost(self, y, y_hat):
        return (-1/self.mini_batch_size) * np.sum(np.nan_to_num(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    # Soft-max function.
    def soft_max(self, X):
        exp = np.exp(X)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        return probabilities

    # Converts each row of the y_hat prediction into a single digit value.
    # The index that gets returned is the one with the highest probability from the soft_max() function.
    def convert(self, Y_hat):
        probabilities = self.soft_max(Y_hat)
        return np.argmax(probabilities, axis=1)

    """
    # Compares the values of Y with Y_hat, and returns the accuracy as the percentage of correct classifications.
    # Is used for debugging and gathering accuracy data.
    # Not used in the final program because TrainDigitX2.csv.gz does not have a label set to compare with.
    def get_accuracy(self, Output, Test_Y):
        wrong_count = 0
        total = Output.shape[0]
        for i in range(Output.shape[0]):
            if Output[i] != Test_Y[i]:
                wrong_count += 1
        # Prints out accuracy statistics.
        correct = total - wrong_count
        print("Correct: " + str(correct))
        print("Incorrect: " + str(wrong_count))
        print(str(correct/total*100) + "% correct.")
        print(" ")
        return correct/total*100
    """

# Creates a Neural_Net object, and sets the epochs, mini_batch_size, learning_rate, and the bias_input values.
print("~~~ Python Neural Network ~~~")
NN = Neural_Net()

epochs = 30
mini_batch_size = 20
learning_rate = 3.0
bias_num = 0.0

print("Epochs: " + str(epochs) + ", mini_batch_size: " + str(mini_batch_size) +
       ", learn rate: " + str(learning_rate) + ", bias_num: " + str(bias_num))

# Trains the Neural_Network, then forward propagates the TestDigitX data through network.
# Then classifies the y_hat output and converts it to the same format of TestDigitY (50,000 rows, 1 col).
NN.train(epochs, mini_batch_size, learning_rate, bias_num)

print("Testing '" + NN.Test_Set + "'.")
Test_Y_hat = NN.fwd_prop(NN.Test_X)
Output = NN.convert(Test_Y_hat)
print("Testing completed.")

# Saves Test_Y_hat to a file specified by the 'Test_Set_Prediction' command-line argument.
# File is formatted in the same way as TrainDigitY.csv (x rows, 1 column). File is stored in 'csv-output' directory.
print("Saving predictions for '" + NN.Test_Set + "' as '" + NN.Test_Set_Prediction + "'.")
np.savetxt(NN.Test_Set_Prediction, Output, fmt='%d')
print(" ")

