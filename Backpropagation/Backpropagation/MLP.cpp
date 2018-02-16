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
    auto i = 0;
    for (auto& l : layers)
    {
        std::cout << "Layer: " << i << std::endl;
        std::cout << "Inputs: " << l.numberOfInputs() << " Outputs: " << l.numberOfOutputs() << '\n';
        for (auto& w : l.getWeights())
        {
            std::cout << w << ' ';
        }
        std::cout << '\n';
        i++;
    }
}

const std::vector<float>& MLP::predict(const std::vector<float>& inputs)
{
    // Buf is a pointer to the most recent output vector. Buf can be passed
    // When a neuron fires
    auto* buf = &inputs;
    for (auto i = 0; i < layers.size(); i++) {
        buf = layers[i].fire(*buf);
    }
    
    return layers.back().getOutputs();
}

void MLP::train(
   const std::vector< std::vector<float> >& X,
   const std::vector< std::vector<int> >& d)
{
    if (X.size() != d.size())
    {
        std::cout << "Input vector is not the same size as the output vector\n";
        throw -1;
    }
    
    auto lastTrainingIndex = (int)(0.8 * X.size());
    for (auto epoch = 0; epoch < epochs; epoch++)
    {
        float rval[2];
        batchUpdate(X, d, lastTrainingIndex);
        validateModel(X, d, lastTrainingIndex + 1, rval);
        auto accuracy = rval[0];
        auto error = rval[1];
        std::cout << "Epoch: " << epoch << " Accuracy: " << accuracy << " Error: " << error << '\n';
    }
}

void
MLP::batchUpdate(const std::vector< std::vector<float> >& X,
                 const std::vector< std::vector<int> >& d,
                 int lastTrainingIndex)
{
    // Apply momentum based on the stored previous dWeights
    for (auto& l : layers)
    {
        l.applyMomentum();
    }
    
    auto i = 0;
    while (i < lastTrainingIndex)
    {
        // Predict will update the internal previous output values
        // in each layer. This prevents needless recomputation of
        // input values for each layer during backpropagation.
        predict(X[i]);
        auto& expected = d[i];
        
        // Adjust output layer
        auto lastIndex = layers.size() - 1;
        layers[lastIndex].adjustAsOutputLayer(expected, layers[lastIndex - 1].getOutputs());
        
        // Adjust hidden nodes
        for (auto j = layers.size() - 2; j > 0; j--)
        {
            auto& inputs = layers[j - 1].getOutputs();
            layers[j].adjustAsHiddenLayer(layers[j + 1], inputs);
            
        }
        
        // Adjust hidden layer
        layers[0].adjustAsHiddenLayer(layers[1], X[i]);
        
        i++;
        if (i % batchSize == 0)
        {
            for (auto& l : layers)
            {
                l.updateWeights();
            }
        }
    }
    if (i % batchSize != 0)
    {
        // Update weights, because the loop ended and they did
        // not get to properly update
        for (auto& l : layers)
        {
            l.updateWeights();
        }
    }
}

inline void
MLP::validateModel(const std::vector< std::vector<float> >& X,
                   const std::vector< std::vector<int> >& d,
                   int firstValidationIndex,
                   float* rval)
{
    auto accuracy = 0.0;
    auto error = 0.0;
    for (auto i = firstValidationIndex; i < X.size(); i++)
    {
        auto& input = X[i];
        auto predicted = predict(input);
        auto& expected = d[i];
        if (verbose)
        {
            std::cout << "Inputs: " << input << std::endl;
            std::cout << "Predicted: " << predicted << std::endl;
            std::cout << "Expected: " << expected << std::endl;
            std::cout << std::endl;
        }
        accuracy += argmax(predicted) == argmax(expected);
        
        for (auto i = 0; i < expected.size(); i++)
        {
            error += mse(expected[i], predicted[i], epsilon);
        }
    }
    
    rval[0] = accuracy /= (X.size() - firstValidationIndex);
    rval[1] = error;
}



MLPLayer::MLPLayer(int _numInputs,
                   int _numOutputs,
                   float _learningRate,
                   float _momentum) :
    numInputs(_numInputs + 1),
    numOutputs(_numOutputs),
    learningRate(_learningRate),
    momentum(_momentum),
    weights(numInputs * _numOutputs, 0),
    outputs(_numOutputs, 0),
    deltas(_numOutputs),
    dWeights(weights.size(), 0)
{

    for (auto i = 0; i < weights.size(); i++) {
        // Set initial weights to random values between -1.0 to 1.0
        weights[i] =  (rand() % 101)/100.0 - 1;
    }
    
}

/*
 * If the model looks like this:
 *    o0        o1
 * w0 w1 w2  w3 w4 w5
 *     x0 x1 x2
 * Num inputs = 3, num outputs = 2
 *
 * Then, delta0 = (d0 - o0^2) * o0^2 * (1 - o0^2)
 * Then, delta1 = (d1 - o1^2) * o1^2 * (1 - o1^2)
 *
 * dw0 = c * delta0 * x0
 * dw1 = c * delta0 * x1
 * dw2 = c * delta0 * x2
 *
 * dw3 = c * delta0 * x0
 * dw4 = c * delta0 * x1
 * dw5 = c * delta0 * x2
 */
void
MLPLayer::adjustAsOutputLayer(const std::vector<int>& expected,
                              const std::vector<float>& inputs)
{
    for (auto i = 0; i < numOutputs; i++)
    {
        // The predicted outputs don't need to be passed in because
        // each layer keeps track of it's expected output.
        auto y = outputs[i];
        auto d = expected[i];
        
        // This is equivalent to (d - y) * f'(a)
        // if f is the sigmoid function.
        auto delta = (d - y) * y * (1 - y);
        deltas[i] = delta;  //  Store delta to reduce previous layer
        
        for (auto j = 0; j < numInputs - 1; j++)
        {
            dWeights[i * numInputs + j] += learningRate * deltas[i] * inputs[j];
        }
        // Adjust bias node
        dWeights[i * numInputs + numInputs - 1] += learningRate * deltas[i];
    }
}

/*
 * Assume network is:
 *
 *    o0'               o1'
 *  w0' w1' w2'    w3' w4' w5'
 *      o0      o1       o2
 *    w0  w1   w2  w3   w4  w5
 *           x0  x1
 *
 *  delta0 = delta0' * w0'  + delta1' * w3'
 */
void
MLPLayer::adjustAsHiddenLayer(const MLPLayer& nextLayer,
                              const std::vector<float>& inputs)
{
    // Plus one because the next layer will have an implicit bias node
    if (numOutputs + 1 != nextLayer.numberOfInputs())
    {
        std::cout << "Number of outputs in next"
                  << " layer is not equal to number of inputs." << std::endl;
        throw -1;
    }
    
    if (numInputs != inputs.size() + 1)
    {
        std::cout
        << numInputs << " does not match " << inputs.size() << std::endl;
        throw -1;
    }
    
    for (auto i = 0; i < outputs.size(); i++)
    {
        auto deltaL = 0.0;
        for (auto l = 0; l < nextLayer.numberOfOutputs(); l++)
        {
            int index = l * nextLayer.numberOfInputs() + i;
            deltaL += nextLayer.weights[index] * nextLayer.deltas[l];
        }
        
        // Multiply outputs with the delta l * wjl term
        auto y = outputs[i];
        deltas[i] = deltaL * y * (1 - y);

        for (auto j = 0; j < numInputs - 1; j++)
        {
            dWeights[i * numInputs + j]  += learningRate * deltas[i] * inputs[j];
        }
        
        // Apply weight change to bias weight
        dWeights[i * numInputs + numInputs - 1] += learningRate * deltas[i];
    }
}

/*
 * Generates a confusion matrix.
 * Typical confusion matrix is stored in memory as:
 * TN FP
 * FN TP
 * n
 *
 * This is stored as
 * [TN TP FN FP ...]
 */
void MLP::evaluate(const std::vector< std::vector<float> > &X,
                   const std::vector< std::vector<int> > &d)
{
    const auto entries = 4;
    std::vector<int> confusionMatrices(this->numOutputs * entries, 0);
    for (auto i = 0; i < X.size(); i++)
    {
        auto predicted = predict(X[i]);
        auto& expected = d[i];
        
        // Convert predicte value to 1 hot encoding
        auto trueIndex = argmax(predicted);
        std::fill(predicted.begin(), predicted.end(), 0);
        predicted[trueIndex] = 1;
        
        for (auto j = 0; j < predicted.size(); j++)
        {
            auto confusionIndex = j * entries;
            
            // Stored as [... TN, TP, FN, FP ..]
            // + 0 if predicts 0, + 1 if prediction is 1
            // + 2 to that if the prediction is incorrect
            confusionIndex += predicted[j];
            if (predicted[j] != expected[j])
            {
                confusionIndex += 2;
            }
            confusionMatrices[confusionIndex]++;
        }
    }
    
    std::cout << "Results: \n";
    std::cout << "n = " << X.size() << std::endl;
    std::cout << "TN FP\nFN TP\n\n";
    for (auto i = 0; i < numOutputs; i++)
    {
        int index = i * numOutputs;
        auto tn = confusionMatrices[index + 0];
        auto tp = confusionMatrices[index + 1];
        auto fp = confusionMatrices[index + 2];
        auto fn = confusionMatrices[index + 3];
        auto precision = 1.0 * tp/(tp + fp);
        auto recall = 1.0 * tp/(tp + fn);

        std::cout << "Class " << i << std::endl;
        std::cout << tn << " " << fp << std::endl;
        std::cout << fn << " " << tp << std::endl;
        std::cout << "Precision: " << precision << " Recall: ";
        std::cout << recall << "\n\n";
    }
}

void MLPLayer::applyMomentum()
{
    for (auto& dw : dWeights)
    {
        dw *= momentum;
    }
}

void MLPLayer::updateWeights()
{
    for (auto i = 0; i < dWeights.size(); i++)
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
    if (inputs.size() + 1 != numInputs)
    {
        std::cout << "Expected " << numInputs << " Inputs\n";
        std::cout << "Received " << inputs.size() << std::endl;
        throw -1;
    }
    
    for (auto i = 0; i < numOutputs; i++)
    {
        auto sum = 0.0;
        for (auto j = 0; j < numInputs - 1; j++)
        {
            sum += weights[i * numInputs + j] * inputs[j];
        }
        
        // Include bias node (input is always "1")
        sum += weights[i * numInputs + numInputs - 1];
        outputs[i] = sigmoid(sum);
    }
    
    return &outputs;
}

inline float sigmoid(float x)
{
    auto rval =  1.0/(1 + exp(-x));
    return rval;
}

template <typename T>
inline float argmax(const std::vector<T>& v)
{
    auto max = v[0];
    auto maxIndex = 0;
    
    for (auto i = 1; i < v.size(); i++)
    {
        if (v[i] > max)
        {
            max = v[i];
            maxIndex = i;
        }
    }
    
    return maxIndex;
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
    return 0.5 * pow(error, 2);
}
