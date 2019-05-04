# Test the accuracy of the classifications produced by 'neural_network.py'

import numpy as np
import sys

# Compares the values of Y with Y_hat, and returns the accuracy as the percentage of correct classifications.
# Is used for debugging and gathering accuracy data.
# Not used in the final program because TrainDigitX2.csv.gz does not have a label set to compare with.
def print_accuracy(Output, Test_Y):
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


# Default files for demo
test_labels_file = 'csv-files/TestDigitY.csv.gz'
neural_network_labels_file = 'csv-output/PredictDigitY.csv.gz'

# Check for command line arguments
if "-demo" not in sys.argv:
    if len(sys.argv) < 3:
        print("Usage: test_accuracy.py TestLabels.csv.gz NeuralNetworkOutput.csv.gz")
        exit(1)
    else:
        test_labels_file = sys.argv[1]
        neural_network_labels_file = sys.argv[2]

# Read in label files
print("Reading in '" + test_labels_file + "'...")
test_labels = np.loadtxt(test_labels_file)

print("Reading in '" + neural_network_labels_file + "'...")
neural_network_labels = np.loadtxt(neural_network_labels_file)

# Print accuracy percentage
print_accuracy(test_labels, neural_network_labels)
