//
//  main.cpp
//  Backpropagation
//
//  Created by Viraj Bangari on 2018-02-11.
//  Copyright © 2018 Viraj. All rights reserved.
//

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "MulticlassNeuralNetwork.hpp"

using std::string;
using std::cout;

/*
 * Parse the preprocessed data
 */
template <int numInputs>
void parseData(
    std::string filename,
    std::vector< std::vector<float> >* inputs,
    std::vector< std::vector<int> >* outputs)
{
    std::ifstream file;
    file.open(filename);

    std::string strline;
    
    // Skip the header row
    std::getline(file, strline);
    while (std::getline(file, strline))
    {
        std::string buf;
        std::stringstream line(strline);

        // Load the input vector;
        std::vector<float> v(numInputs, 0);
        inputs->push_back(v);
        for (int i  = 0; i < numInputs; i++)
        {
            std::getline(line, buf, ',');
            (*inputs)[inputs->size() - 1][i] = stod(buf);
        }

        // Load an output vector
        std::getline(line, buf);
        auto val = stoi(buf);
        switch (val)
        {
            case 5:
                outputs->push_back({0, 0, 1});
                break;
            case 7:
                outputs->push_back({0, 1, 0});
                break;
            case 8:
                outputs->push_back({1, 0, 0});
                break;
            default:
                cout << "Undefined output value: " << buf;
                throw -1;
        }
    }
    
    file.close();
}

int main(int argc, const char * argv[])
{
    // No calls to C style i/o, so this should improve performance
    std::ios_base::sync_with_stdio(false);

    // Initialize variables
    const auto numInputs = 5;
    const auto numEpochs = 100;
    const auto batchSize = 30;
    const auto epsilon = 0.2;
    std::vector< std::vector<float> > inputs;
    std::vector< std::vector<int> > outputs;

    // Load the training data
    cout << "Loading training data...\n";
    parseData<numInputs>("./data/training.csv", &inputs, &outputs);
    std::cout << "Loaded " << inputs.size();
    std::cout << " inputs and " << outputs.size() << " outputs\n";
    
    // Initial model parameters. Add +1 for the bias
    // Number of inputs, batch size, epochs
    MulticlassNeuralNetwork mnn(numInputs, batchSize, numEpochs, epsilon);
    mnn.addLayer(4, 0.1, 0.1);
    mnn.addLayer(10, 0.1, 0.1);
    mnn.addLayer(3, 0.1, 0.1);

    // Print initial weights
    std::cout << "Initial Weights" << std::endl;
    mnn.printWeights();
    std::cout << std::endl;
    
    // Train the model
    mnn.train(inputs, outputs);
    std::cout << "Weights after training" << std::endl;
    mnn.printWeights();
    std::cout << std::endl;
    
    // Loading testing data
    inputs.clear();
    outputs.clear();
    parseData<numInputs>("./data/testing.csv", &inputs, &outputs);
    
    // Evaluate the model (prints results)
    mnn.evaluate(inputs, outputs);

    return 0;
}
