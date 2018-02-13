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
        inputs[inputs.size() - 1][numInputs] = 1;
        
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

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values)
{
    for (auto const& value : values)
    {
        output << value << " ";
    }
    return output;
}

int main(int argc, const char * argv[])
{
    const int numInputs = 2;
    
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
    MLP mlp(numInputs + 1, 10, 10, 0.2, false);
    mlp.addLayer(2, 0.5, 0.1);
    mlp.addLayer(3, 0.2, 0.4);

    // Print initial weights
    mlp.printWeights();
    mlp.train(inputs, outputs);
    mlp.printWeights();
    
    auto r1 = mlp.predict(inputs[0]);
    std::cout << r1 << std::endl;
    
    auto r2 = mlp.predict(inputs[1]);
    std::cout << r2 << std::endl;
    
    auto r3 = mlp.predict(inputs[3]);
    std::cout << r3 << std::endl;
    
    return 0;
}
