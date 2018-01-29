#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>

/* 
 * Perceptron class for linear seperability
 */
class Perceptron
{
  private:
    const int numInputs;
    const int numOutputs;
    float * weights;

    const float c;
    static const int MAX_ITERATIONS = 1000;
    void updateWeights(float weights[], float input[], int output);
    int predict(float const weightsPtr[], float const input[]);
    
  public:
    Perceptron(int _inputs, int _outputs);
    ~Perceptron()
    {
        delete[] weights;
    }

    /*
     * Prints the weights of the perceptron
     */
    void printWeights();

    /*
     * Updates the internal weight and threshold vectors
     * based on the input value and expected outputs
     */
    void train(const int N, float * const inputs, int * const outputs);
    /*
     * Gives the prediction for all outputs
     */
    void makePrediction(float const input[], int output[]);
};

#endif
