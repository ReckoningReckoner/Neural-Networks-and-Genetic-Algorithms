//
//  MulticlassNeuralNetwork.hpp
//
//  Neural network with a configurable number of layers best suited for
//  multi-class (binary) predictions. Using backpropagation with sigmoid
//  and momentum.
//
//  Created by Viraj Bangari on 2018-02-11.
//  Copyright © 2018 Viraj. All rights reserved.
//

#ifndef MulticlassNeuralNetwork_hpp
#define MulticlassNeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <iostream>

class MNNLayer;
class MulticlassNeuralNetwork;

class MulticlassNeuralNetwork
{
private:
    int numInputs;
    int numOutputs;
    float epsilon;
    int batchSize;
    int epochs;
    bool verbose;
    std::vector<MNNLayer> layers;
    
    void validateModel(
       const std::vector< std::vector<float> >& X,
       const std::vector< std::vector<int> >& d,
       int firstValidationIndex,
       float* rvals);
    
    void batchUpdate(
        const std::vector< std::vector<float> >& X,
        const std::vector< std::vector<int> >& d,
        int lastTrainingIndex);
public:
    MulticlassNeuralNetwork(
        int _numInputs,
        int _batchSize,
        int _epochs,
        float _epsilon,
        bool verbose=false);

    void addLayer(int newNumOutputs, float learningRate, float momentum);
    const std::vector<float>& predict(const std::vector<float>& inputs);
    void train(
        const std::vector< std::vector<float> >& X,
        const std::vector< std::vector<int> >& d);
    void printWeights() const;
    void train(std::vector<float>& X, std::vector<int>& d);
    void evaluate(const std::vector< std::vector<float> >& X,
                  const std::vector< std::vector<int> >& d);
};

/*
 * Layers between MLP.
 * For a single perceptron, the activiation is:
 * a = sum(wi * xi).
 *
 * Each MLP stores its previous output, delta as dWeights.
 * This increases the amound of memory, but should decrease the
 * computation time.
 */
class MNNLayer
{
private:
    const int numInputs;
    const int numOutputs;
    const float learningRate;
    float momentum;
    std::vector<float> weights;
    std::vector<float> outputs;
    std::vector<float> deltas;
    std::vector<float> dWeights;

public:
    MNNLayer(int _numInputs,
             int _numOutputs,
             float _learningRate,
             float _momentum);
    const std::vector<float>* fire(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const
    {
        return outputs;
    }
    const std::vector<float>& getDeltas() const
    {
        return deltas;
    }
    int numberOfInputs() const
    {
        return numInputs;
    }
    int numberOfOutputs() const
    {
        return numOutputs;
    }
    const std::vector<float>& getWeights() const
    {
        return weights;
    }
    /*
     * This method should be called BEFORE the weights
     * updated and summed
     */
    void applyMomentum();
    void adjustAsOutputLayer(const std::vector<int>& expected,
                             const std::vector<float>& inputs);
    void adjustAsHiddenLayer(const MNNLayer& nextLayer,
                             const std::vector<float>& inputs);
    void updateWeights();
};

float sigmoid(float x);
float err(float d, float y, float epsilon=0);
float mse(float d, float y, float epsilon=0);

template <typename T>
float argmax(const std::vector<T>& v);

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values)
{
    for (auto const& value : values)
    {
        output << value << " ";
    }
    return output;
}

#endif /* MulticlassNeuralNetwork_hpp */
