# CMPE452 Assignment 1
## Viraj Bangari 10186046

### Model
The classification of iris setosa, iris versicolor and iris virginica flowers was done using a four-input perceptron. The four inputs directly corresponded to the sepal length, sepal width, petal length and petal width of the flowers. The perceptron has three binary outputs, each corresponding to the species of flower.

```
Plant            Output
Setosa          [1, 0, 0]
Versicolor      [1, 0, 0]
Virginica       [1, 0, 0]
```

A threshold function was used to determine whetehr an output would be 1 or 0. The value of the threshold was the fifth weight (the bias) corresponding to the output neuron. The model used simple-feedback learning with c = 0.4 to adjust the weights. Neurons also adjusteed their threshold values using the same method.

#### Terminating Criteria
For each input and output pair, the sum squared error was calculated. The weights are updated using feedback learning until either the sum squared error is equal to zero, the new sum squared value is greater than the old one + 5 (in which case the old weight vector is used), or if 1000 iterations are reached. This can be seen in line 88 - 96 of Percetron.cpp.

### Results
Note, output 1 is setosa, 2 is versicolor and 3 is virginica.

#### Initial and final weight vector
```
Before Training:
Weights for output 1:
-29, 12, -15, -12, 71
Weights for output 2:
77, 70, 31, 3, -41
Weights for output 3:
36, 62, 34, 77, -40

After Training:
Weights for output 1:
4.68, 34.96, -10.72, -12.2, 82.6
Weights for output 2:
-10.88, 10.04, 0.759995, -4.52, -17
Weights for output 3:
-35.08, 12.88, 14.88, 73.8799, -19.6
```

#### Total sum squared error
```
Sum squared error: 16
```

#### Number of iterations
```
Iterations for output 1: 8
Iterations for output 2: 4
Iterations for output 3: 3
```

#### Predicted values vs actual values
See `trainingOutput.out`. The data corresponds with the `data/train.txt` file.

#### Precision and cecall
Iris setosa classification
Precision: 100, Recall: 100

Iris versicolor classification
Precision: 95, Recall: 65.5172

Iris virginica classification
Precision: 75, Recall: 100


### Third party library
The perceptron module in scikit-learn for python3 was used. It can be run using Python 3.6.
The source code can be found in `Perceptron.py`. The precision and recall values are:

Iris setosa classification
Precision: 100, Recall: 100

Iris versicolor classification
Precision: 33.33, Recall: 100

Iris virginica classification
Precision: 100, Recall: 60

The sklearn perceptron has similar results to the model I used. The important part is that it achieved 100% precision and recall, which should theoretically occur since it is linearly serparable.

## Instructions
The neural network is contained the `Perceptron.cpp` and `Perceptron.h` files. It can be compiled using `make`. The compiler I used was clang++ for x86_64-apple-darwin16.7.0. The code should compile well using G++ on most windows/Linux environments. Running the executible will print out the weights, number of iterations, precision, recall and error squared. Output 1 corresponds to setosa, 2 with versicolor, and 3 with virginica.