/*
* CMPE452 Perceptron Assignment 1
* Viraj Bangari
* January 2020
*
* Implementation and usage of a simple single-layer perceptron
* that uses the iris dataset.
*/

#include "Perceptron.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>  


Perceptron::Perceptron(int _inputs, int _outputs) : 
    numInputs(_inputs), numOutputs(_outputs), c(0.4)
{
    srand(0x8c23); // Seed for RNG.

    // Store weights matrix as a flat array. Add plus 1
    // for the bias weight.
    weights = new float[(numInputs + 1) * numOutputs];

    for (int i = 0; i < (numInputs + 1) * numOutputs; i++)
    {
        int factor = rand() % 2 ? -1 : 1;
        weights[i] = factor * (rand() % 100);
    }
}

void Perceptron::printWeights()
{
    /*
     * Prints the weights of the perceptron
     */
    for (int i = 0; i < numOutputs; i++)
    {
        std::cout << "Weights for output: " << i + 1 << '\n';
        float* weightPtr = weights + i * (numInputs + 1);
        for (int j = 0; j < (numInputs + 1); j++)
        {
            std::cout << weightPtr[j];
            if (j != numInputs)
            {
                std::cout << ", ";
            }
        }
        std::cout << '\n';
    }
}

void Perceptron::train(const int N, float * const inputs, int* const outputs)
{
    /*
     * Trains the perceptron using simple-feedback learning. This essentially means
     * adjusting the weight vector.
     *
     * Training is repeatedly done until either the error squared value is 0, the
     * new error squared value is greater than the old error squared + 5 or a max number
     * of iterations is reached.
     */
    const int errorThreshold = 5;

    for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++)
    {
        int numIterations = 0;
        int oldErrorSquared = N;
        float * weightsPtr = weights + outputIndex * (numInputs + 1);
        float * oldWeights = new float[numInputs + 1];

        while (numIterations < MAX_ITERATIONS)
        {
            // Back up the old weights
            std::memcpy(oldWeights, weightsPtr, numInputs + 1);

            // Update the weights by checking over every single input and output.
            for (int n = 0; n < N; n++)
            {
                float * nthInput = inputs + n * numInputs;
                int * nthOutput = outputs + n * numOutputs;
                updateWeights(weightsPtr, nthInput, nthOutput[outputIndex]);
            }

            int errorSq = 0;
            for (int n = 0; n < N; n++)
            {
                float * nthInput = inputs + n * numInputs;
                int expected = outputs[n * numOutputs + outputIndex];
                int prediction = predict(weightsPtr, nthInput);
                errorSq += pow(prediction - expected, 2);
            }

            if (errorSq == 0)
            {
                break;
            }
            else if (oldErrorSquared + errorThreshold < errorSq)
            {
                std::memcpy(weightsPtr, oldWeights, numInputs + 1);
                break;
            }

            oldErrorSquared = errorSq;
            numIterations++;
        }

        delete[] oldWeights;
        std::cout << "Iterations for output " << outputIndex << ": " << numIterations << std::endl;
    }
}

int Perceptron::predict(float const weightsPtr[], float const input[])
{
    // Get the sum of the weighted inputs
    float a = 0;
    for (int inputIndex = 0; inputIndex < numInputs; inputIndex++)
    {
        float weight = weightsPtr[inputIndex];
        float xi = input[inputIndex];
        a += weight * xi;
    }
    // Add the bias weight
    a -= weightsPtr[numInputs];
    return a >= 0;
}

void Perceptron::makePrediction(float const input[], int output[])
{
    for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++)
    {
        float* weightsPtr = weights + outputIndex * (numInputs + 1);
        int predictedOutput = predict(weightsPtr, input);
        output[outputIndex] = predictedOutput;
    }
}

void Perceptron::updateWeights(
    float weightsPtr[],
    float input[],
    int expectedOutput)
{
        // Get the sum of the weighted inputs
        float a = 0;
        for (int inputIndex = 0; inputIndex < numInputs; inputIndex++)
        {
            float weight = weightsPtr[inputIndex];
            float xi = input[inputIndex];
            a += weight * xi;
        }
        // Add the bias weight
        a -= weightsPtr[numInputs];
        int predictedOutput = a >= 0;

        // Apply simple feedback if the predicted does not match the expected
        if (predictedOutput != expectedOutput)
        {
            int factor = predictedOutput > expectedOutput ? -1 : 1; 
            for (int inputIndex = 0; inputIndex < numInputs; inputIndex++)
            {
                float xi = input[inputIndex];
                weightsPtr[inputIndex] += factor * c * xi;
            }
            weightsPtr[numInputs] += c;
        }
}

void mapOutputToArray(std::string& output, int outvector[])
{
    outvector[0] = 0;
    outvector[1] = 0;
    outvector[2] = 0;

    if (output == "Iris-setosa")
    {
        outvector[0] = 1;
    }
    else if (output == "Iris-versicolor")
    {
        outvector[1] = 1;
    }
    else if (output == "Iris-virginica")
    {
        outvector[2] = 1;
    }
    else {
        std::cout << "Invalid output: \'" << output << '\'' << std::endl;
        throw output;
    }
}

template < int bufsize, int numInputs, int numOutputs >
int loadData(
    std::string filename, 
    float inputs[bufsize][numInputs], 
    int outputs[bufsize][numOutputs])
{
    int numTrainingPoints = 0;
    std::ifstream file(filename);
    std::string line;
    char tmp[50];

    while (std::getline(file, line))
    {
        int tmpIndex = 0;
        int inputIndex = 0;
        for (int i = 0; line[i] != '\n' && line[i]; i++)
        {
            if (line[i] != ',')
            {
                tmp[tmpIndex] = line[i];
                tmpIndex++;
            }
            else 
            {
                tmp[tmpIndex] = '\0';
                inputs[numTrainingPoints][inputIndex] = atof(tmp);
                inputIndex++;
                tmpIndex = 0;
            }
        }
        tmp[tmpIndex] = '\0';
        std::string str(tmp);
        mapOutputToArray(str, outputs[numTrainingPoints]);
        numTrainingPoints++;
    }
    file.close();

    return numTrainingPoints;
}

int main()
{
    const int bufsize = 128;
    const int numInputs = 4;
    const int numOutputs = 3;
    float inputs[bufsize][numInputs];
    int outputs[bufsize][numOutputs];
    Perceptron p(numInputs, numOutputs);

    // Display randomly assigned weights
    std::cout << "Before Training:\n";
    p.printWeights();
    std::cout << "\n";

    // Read data from text file and load into memory
    int numTrainingPoints = loadData<bufsize, numInputs, numOutputs>("data/train.txt", inputs, outputs);

    // Train the model
    p.train(numTrainingPoints, (float*)inputs, (int*)outputs);
    std::cout << "\nAfter Training:\n";
    p.printWeights();
    std::cout << "\n";

    
    // Compute Statistics
    // Matrix is:
    // | FN | TN |
    // | FP | TP |
    int relevanceMatrix[3][2][2];
    int numTestingPoints = loadData<bufsize, numInputs, numOutputs>("data/test.txt", inputs, outputs);
    int error = 0;

    std::ofstream outfile;
    outfile.open("trainingOutput.out", std::ofstream::trunc|std::ofstream::out);
    for (int n = 0; n < numTestingPoints; n++)
    {
        int predictedOutputs[numOutputs];
        int * expectedOutputs = outputs[n];

        p.makePrediction(inputs[n], predictedOutputs);
        for (int i = 0; i < 3; i++)
        {
            int d = predictedOutputs[i] - expectedOutputs[i];
            error += d * d;
            relevanceMatrix[i][d == 0][predictedOutputs[i]]++;
        }

        outfile << "Predicted: ";
        for (int i = 0; i < numOutputs; i++)
            outfile << predictedOutputs[i];
        outfile << '\n';

        outfile << "Actual:    ";
        for (int i = 0; i < numOutputs; i++)
            outfile << outputs[n][i];
        outfile << '\n';
    }

    for (int i = 0; i < 3; i++) {
        float tp = relevanceMatrix[i][1][0];
        float fn = relevanceMatrix[i][0][0];
        float fp = relevanceMatrix[i][0][1];

        float precision = tp/(tp + fp);
        float recall = tp/(tp + fn);

        std::cout << "Precision: " << precision * 100 << ", ";
        std::cout << "Recall: " << recall * 100 << std::endl;
    }

    std::cout << "Sum squared error: " << error << '\n';


    return 0;
}
