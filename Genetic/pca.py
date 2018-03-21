"""
CMPE 452 Assignment 3
Viraj Bangari

Performs PCA on an audio sample of opera speaker and voice.

Henlo TA hope you are having a great day so far.
"""

import numpy as np
import scipy as sp
from scipy.io import wavfile


def performPCA(X):
    # PCA parameters
    dimensionality = 1
    learningRate = 0.1
    maxIterations = 10
    epsilon = 0.01

    print("Learning rate: {}".format(learningRate))
    print("Epsilon: {}".format(epsilon))

    # Randomly assign weights
    weights = np.array([[1.0, 0.0]])
    print("Initial weights:\n{}".format(weights))

    # Perform PCA
    for i in range(maxIterations):
        for x in X:
            y = np.matmul(weights, x)
            delta_w = learningRate * (y * x - y**2 * weights)
            weights += delta_w

        if abs(np.linalg.norm(weights) - 1) < epsilon:
            print("Finished after {} iterations".format(i + 1))
            print("Final weights:\n{}".format(weights))
            break

    # Create output wavefiles
    output = np.zeros((X.shape[0], dimensionality))
    for i in range(X.shape[0]):
        output[i] = np.matmul(weights, X[i])

    return output


if __name__ == "__main__":
    # Data file
    data = "./Data/sound.csv"
    samplingFreq = 8000

    # Load data into np matrix
    X = np.genfromtxt(data, delimiter=',')

    print("Performing PCA for first component")
    output = performPCA(X)

    # Oh shit this actually worked!
    wavfile.write("Output/PCA.wav", samplingFreq, output)
    np.savetxt("Output/PCA.csv", output)
