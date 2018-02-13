//
//  MLP.hpp
//  Backpropagation
//
//  Created by Viraj Bangari on 2018-02-11.
//  Copyright © 2018 Viraj. All rights reserved.
//

#ifndef MLP_hpp
#define MLP_hpp

#include <stdio.h>
#include <vector>
#include <iostream>

class MLPLayer;
class MLP;

class MLP
{
private:
    int numInputs;
    int numOutputs;
    float epsilon;
    int batchSize;
    int epochs;
    bool verbose;
    std::vector<MLPLayer> layers;
    
    float validateModel(
       std::vector< std::vector<float> >& X,
       std::vector< std::vector<int> >& d,
       int firstValidationIndex,
       float* percentageCorrect);
    
    void batchUpdate(
        std::vector< std::vector<float> >& X,
        std::vector< std::vector<int> >& d,
        int lastTrainingIndex);
public:
    MLP(
        int _numInputs,
        int _batchSize,
        int _epochs,
        float _epsilon,
        bool verbose=false);

    void addLayer(int newNumOutputs, float learningRate, float momentum);
    std::vector<float> predict(const std::vector<float>& inputs);
    void train(
        std::vector< std::vector<float> >& X,
        std::vector< std::vector<int> >& d);
    void printWeights() const;
    void train(std::vector<float>& X, std::vector<int>& d);
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
class MLPLayer
{
private:
    const int numInputs;
    const int numOutputs;
    const float learningRate;
    const float momentum;
    std::vector<float> weights;
    std::vector<float> outputs;
    std::vector<float> deltas;
    std::vector<float> dWeights;

public:
    MLPLayer(int _numInputs,
             int _numOutputs,
             float _learningRate,
             float _momentum);
    const std::vector<float>* fire(const std::vector<float>& inputs);
    const std::vector<float>& getPreviousOutputs() const
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
    void adjustAsHiddenLayer(const MLPLayer& nextLayer,
                             const std::vector<float>& inputs);
    void updateWeights();
};

float sigmoid(float x);
float err(float d, float y, float epsilon=0);
float mse(float d, float y, float epsilon=0);

#endif /* MLP_hpp */
