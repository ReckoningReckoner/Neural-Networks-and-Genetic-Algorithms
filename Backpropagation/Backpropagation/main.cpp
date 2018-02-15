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

#include "MLP.hpp"

using std::string;
using std::cout;

/*
 * Parse the preprocessed data
 */
template <int numInputs>
void parseData(
    std::string filename,
    std::vector< std::vector<float> >& inputs,
    std::vector< std::vector<int> >& outputs)
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
        inputs.push_back(v);
        for (int i  = 0; i < numInputs; i++)
        {
            std::getline(line, buf, ',');
            inputs[inputs.size() - 1][i] = stod(buf);
        }

        // Load an output vector
        std::getline(line, buf);
        auto val = stoi(buf);
        switch (val)
        {
            case 5:
                outputs.push_back({0, 0, 1});
                break;
            case 7:
                outputs.push_back({0, 1, 0});
                break;
            case 8:
                outputs.push_back({1, 0, 0});
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
    const auto numInputs = 4;
    
    // Load the training data
    cout << "Loading training data...\n";
    std::vector< std::vector<float> > inputs;
    std::vector< std::vector<int> > outputs;

    // In the training data, the csv has 5 input values and one output value.
    parseData<numInputs>("./data/training.csv", inputs, outputs);
    std::cout << "Loaded " << inputs.size()
              << " inputs and " << outputs.size() << " outputs\n";
    
    // Initial model parameters. Add +1 for the bias
    // Number of inputs, batch size, epochs
    MLP mlp(numInputs, 30, 20, 0.2, false);
    mlp.addLayer(6, 0.1, 0.0);
    mlp.addLayer(3, 0.1, 0.0);

    // Print initial weights
    mlp.printWeights();
    std::cout << std::endl;
    
    mlp.train(inputs, outputs);
    mlp.printWeights();
    
    return 0;
}
