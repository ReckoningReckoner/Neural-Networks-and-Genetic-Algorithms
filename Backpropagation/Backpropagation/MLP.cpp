//
//  MLP.cpp
//  Backpropagation
//
//  Created by Viraj Bangari on 2018-02-11.
//  Copyright © 2018 Viraj. All rights reserved.
//

#include "MLP.hpp"
#include <math.h>

MLP::MLP(
    int _numInputs,
    int _batchSize,
    int _epochs,
    float _epsilon,
    bool _verbose) :
        numInputs(_numInputs),
        batchSize(_batchSize),
        epochs(_epochs),
        // Set numOutputs to numInputs since there are no layers yet.
        numOutputs(numInputs),
        epsilon(_epsilon),
        verbose(_verbose)
{
    srand(0);
};

void MLP::addLayer(int newNumOutputs, float learningRate, float momentum)
{
    // TODO: Use heap if this is too slow
    MLPLayer newLayer(numOutputs, newNumOutputs, learningRate, momentum);
    layers.push_back(newLayer);
    numOutputs = newNumOutputs;
}

void MLP::printWeights() const {
    int i = 0;
    for (const MLPLayer& l : layers)
    {
        std::cout << "Layer: " << i << std::endl;
        std::cout << "Inputs: " << l.numberOfInputs() << " Outputs: " << l.numberOfOutputs() << '\n';
        for (const float& w : l.getWeights())
        {
            std::cout << w << ' ';
        }
        std::cout << '\n';
        i++;
    }
}

std::vector<float> MLP::predict(const std::vector<float>& inputs)
{
    // Buf is a pointer to the most recent output vector. Buf can be passed
    // When a neuron fires
    const std::vector<float>* buf = &inputs;
    
    for (int i = 0; i < layers.size(); i++) {
        buf = layers[i].fire(*buf);
    }
    
    return *buf;
}

void MLP::train(
   std::vector< std::vector<float> >& X,
   std::vector< std::vector<int> >& d)
{
    if (X.size() != d.size())
    {
        std::cout << "Input vector is not the same size as the output vector\n";
        throw -1;
    }
    
    int lastTrainingIndex = (int)(0.8 * X.size());
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float percentageCorrect = 0;
        batchUpdate(X, d, lastTrainingIndex);
        float error = validateModel(X, d, lastTrainingIndex + 1, &percentageCorrect);
        std::cout << "Epoch: " << epoch << ", Error: " << error << '\n';
        std::cout << "Epoch: " << epoch << ", % correct: " << percentageCorrect << '\n';
    }
}

void
MLP::batchUpdate(std::vector< std::vector<float> >& X,
                 std::vector< std::vector<int> >& d,
                 int lastTrainingIndex)
{
    // Apply momentum based on the stored previous dWeights
    for (MLPLayer& l : layers)
    {
        l.applyMomentum();
    }
    
    int i = 0;
    while (i < lastTrainingIndex)
    {
        // Predict will update the internal previous output values
        // in each layer. This prevents needless recomputation of
        // input values for each layer during backpropagation.
        predict(X[i]);
        std::vector<int>& expected = d[i];
        
        // Adjust output layer
        // TODO: This will not work if the MLP has no hidden layers
        layers[layers.size() - 1]
            .adjustAsOutputLayer(expected,
                                 layers[layers.size() - 2]
                                     .getPreviousOutputs());
        
        // Adjust hidden nodes
        for (unsigned long j = layers.size() - 2; j > 0; j--)
        {
            const std::vector<float>& inputs =
                    layers[j - 1].getPreviousOutputs();
            layers[j]
                .adjustAsHiddenLayer(layers[j + 1], inputs);
            
        }
        
        // Adjust hidden layer
        layers[0].adjustAsHiddenLayer(layers[1], X[i]);
        
        i++;
        if (i % batchSize == 0)
        {
            for (MLPLayer& l : layers)
            {
                l.updateWeights();
            }
        }
    }
    if (i % batchSize != 0)
    {
        // Update weights, because the loop ended and they did
        // not get to properly update
        for (MLPLayer& l : layers)
        {
            l.updateWeights();
        }
    }
}

inline float
MLP::validateModel(std::vector< std::vector<float> >& X,
                   std::vector< std::vector<int> >& d,
                   int firstValidationIndex,
                   float* percentageCorrect)
{
    float error = 0;
    for (int i = firstValidationIndex; i < X.size(); i++)
    {
        std::vector<float> predicted = predict(X[i]);
        std::vector<int>& expected = d[i];
        
        float maxval = 0;
        int maxIndex = 0;
        for (int j = 0; j < numOutputs; j++)
        {
            error += mse(expected[j], predicted[j]);
            if (expected[j] > maxval)
            {
                maxval = expected[j];
                maxIndex = j;
            }
        }
        
        if (predicted[maxIndex] == 1)
        {
            *percentageCorrect += 1;
        }
    }
    
    return error;
}



MLPLayer::MLPLayer(int _numInputs,
                   int _numOutputs,
                   float _learningRate,
                   float _momentum) :
    numInputs(_numInputs),
    numOutputs(_numOutputs),
    learningRate(_learningRate),
    momentum(_momentum),
    weights(_numInputs * _numOutputs, 0),
    outputs(_numOutputs, 0),
    deltas(_numOutputs),
    dWeights(weights.size(), 0)
{

    for (int i = 0; i < weights.size(); i++) {
        // Set initial weights to random values between -1.0 to 1.0
        weights[i] = (rand() % 101)/100.0 - 1;
    }
    
}

void
MLPLayer::adjustAsOutputLayer(const std::vector<int>& expected,
                              const std::vector<float>& inputs)
{
    for (int i = 0; i < numOutputs; i++)
    {
        // The predicted outputs don't need to be passed in because
        // each layer keeps track of it's expected output.
        float y = outputs[i];
        float d = expected[i];
        
        // This is equivalent to (d - y) * f'(a)
        // if f is the sigmoid function.
        float delta = (d - y * y) * y * y * (1 - y * y);
        deltas[i] = delta;
        for (int j = 0; j < numInputs; j++)
        {
            int index = i * numInputs + j;
            dWeights.at(index) +=
                    learningRate * deltas[i] * inputs[j];
        }
    }
}

void
MLPLayer::adjustAsHiddenLayer(const MLPLayer& nextLayer,
                              const std::vector<float>& inputs)
{
    if (numOutputs != nextLayer.numberOfInputs())
    {
        std::cout << "Number of outputs in next"
                  << " layer is not equal to number of inputs." << std::endl;
        throw -1;
    }
    
    if (numInputs != inputs.size())
    {
        std::cout
        << numInputs << " does not match " << inputs.size() << std::endl;
        throw -1;
    }
    
    for (int i = 0; i < outputs.size(); i++)
    {
        auto deltaL = 0.0;
        for (int j = 0; j < nextLayer.numberOfOutputs(); j++)
        {
            int index = j * nextLayer.numberOfInputs() + i;
            deltaL += nextLayer.weights.at(index) * nextLayer.deltas.at(j);
        }
        auto y = outputs[i];
        deltas[i] = deltaL * y * y * (1 - y * y);

        for (int j = 0; j < numInputs; j++)
        {
            int index = i * numInputs + j;
            dWeights.at(index)  += learningRate * deltas[i] * inputs[j];
        }
    }
}

void MLPLayer::applyMomentum()
{
    for (float& dw : dWeights)
    {
        dw *= momentum;
    }
}

void MLPLayer::updateWeights()
{
    for (int i = 0; i < dWeights.size(); i++)
    {
        weights[i] += dWeights[i];
    }
}

/*
 * If inputs are x0, x1, x2
 * and weights are:
 *   w0 w1 w2
 *   w3 w4 w5
 *
 * Then the outputs are:
 * o0 = x0 w0 + x1 w1 + x2 w2
 * o1 = x0 w3 + x1 w4 + x2 w5
 *
 * In this case, numInputs would be 3, num outputs are 2.
 *
 * numInputs is the number of columns, numOutputs is the number of
 * rows.
 */
const std::vector<float>* MLPLayer::fire(const std::vector<float>& inputs)
{
    if (inputs.size() != numInputs)
    {
        std::cout << "Expected " << numInputs << " Inputs\n";
        std::cout << "Received" << inputs.size() << std::endl;
        throw -1;
    }
    
    for (int i = 0; i < numOutputs; i++)
    {
        float sum = 0;
        for (int j = 0; j < numInputs; j++)
        {
            // This should be equivalent to w[i][j]
            int index = i * numInputs + j;
            sum += weights.at(index) * inputs[j];
        }
        outputs[i] = sigmoid(sum);
    }
    
    return &outputs;
}

inline float sigmoid(float x)
{
    float rval =  1/(1 + exp(-x));
    return rval;
}

inline float err(float d, float y, float epsilon)
{
    if (fabs(d - y) <= epsilon)
    {
        return 0;
    }
    return d - y ;
}

inline float mse(float d, float y, float epsilon)
{
    float error = err(d, y, epsilon);
    if (error == 0)
    {
        return 0;
    }
    return pow(error, 2);
}
