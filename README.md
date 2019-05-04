# NeuralNetwork_DigitRecognition
Simple neural network to classify MNIST images of handwritten digits

## Requirements
* Python 3.x
* Numpy Python Library

## Running the Neural Network Demo
1. Execute `python neural_network.py -demo` to run a demo neural network. This network will use the supplied MNIST hand-written digits as inputs.
2. After the program has finished, an output set of classifications *csv-output/PredictDigitY.csv.gz* will be produced by the neural network.

## Testing the Accuracy of the Neural Network Demo Classifications
1. Execute `python test_accuracy.py -demo` to compare the classifications produced by the neural network with the true classifications in *csv-files/TestDigitY.csv.gz*.
2. An accuracy percentage will be printed out.
