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
    while (std::getline(file, strline))
    {
        std::stringstream line(strline);
        std::string buf;
        
        // Load the input vector;
        std::vector<float> v(numInputs + 1, 0);
        inputs.push_back(v);
        for (int i  = 0; i < numInputs; i++)
        {
            std::getline(line, buf, ',');
            inputs[inputs.size() - 1][i] = stod(buf);
        }
        inputs[inputs.size() - 1][numInputs] = 1;  // Add bias input

        // Load an output vector
        std::getline(line, buf);
        int val = stoi(buf);
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
    // Load the training data
    cout << "Loading training data...\n";
    std::vector< std::vector<float> > inputs;
    std::vector< std::vector<int> > outputs;

    const int numInputs = 5;
    // In the training data, the csv has 5 input values and one output value.
    parseData<numInputs>("./data/training.csv", inputs, outputs);
    std::cout << "Loaded " << inputs.size()
              << " inputs and " << outputs.size() << " outputs\n";
    
    // Initial model parameters. Add +1 for the bias
    // Number of inputs, batch size, epochs
    MLP mlp(numInputs + 1, 500, 1, 0);
    mlp.addLayer(6, 0.1, 0.5);
    mlp.addLayer(3, 0.1, 0.5);

    // Print initial weights
    mlp.printWeights();
    mlp.train(inputs, outputs);
    
//    std::vector<float> results = mlp.predict(inputs[0]);
//    for (float& r : results)
//    {
//        std::cout << r << ' ';
//    }
//    std::cout << '\n';

  
    return 0;
}
