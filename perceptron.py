# Perceptron Learning in Python

import random

InputVectorSize = 2 # Perceptron will have two inputs
WeightSize = InputVectorSize + 1  # weights + bias 
LEARNING_RATE = 0.1 # Multiple this by the error (output - desiredOutput) to change weights per iteration 
ITERATIONS = 10 # Could be called max iterations

weights = [0.0] * WeightSize # Create a list of weights initiaized to 0

def initialize():
    global weights # global as the weights variable is defined globally
    # Initialize the weights with random values. Range function used to build out random values in list weights
    for i in range(WeightSize): 
        weights[i] = random.random()

def feedforward(InputVectorSizes):
    # Function to sum the inputs multiplies by their weights then to add the bias weight as well
    sum = 0.0
    # Calculate InputVectorSizes * weights
    for i in range(InputVectorSize):
        sum = sum + (weights[i] * InputVectorSizes[i])
    # Add in the bias
    sum = sum + weights[InputVectorSize]
    # Activation function (1 if value >= 1.0)
    return 1 if sum >= 1.0 else 0

def train():
    global weights
    iterations = 0
    iteration_error = 0

    # Train the boolean OR set
    test = [[0, 0], [0, 1], [1, 0], [1, 1]]

    while True:
        iteration_error = 0

        print(f"Iteration {iterations}")

        for InputVectorSizes in test:
            desired_output = InputVectorSizes[0] or InputVectorSizes[1] # To train to OR vs and simply change and to or in this line and vice versa
            output = feedforward(InputVectorSizes) # Call Feed Forward to get output of algorithm as it 
            error = desired_output - output # Compute the error by taking the desired output and subtracting the actual output

            print(f"{InputVectorSizes[0]} or {InputVectorSizes[1]} = {output} ({desired_output})")

            # Update weights
            weights[0] += LEARNING_RATE * error * InputVectorSizes[0] # Highlight the input weights are the weights plus the learning rate # error * input
            weights[1] += LEARNING_RATE * error * InputVectorSizes[1]
            weights[2] += LEARNING_RATE * error

            iteration_error += error ** 2 # Throughing an iteration error of 1 even if it worked correctly in some cases

        print(f"Iteration error {iteration_error}\n")

        if iteration_error <= 0 or iterations >= ITERATIONS:
            break

        iterations += 1

def main():
    initialize()
    train()
    print(f"Final weights {weights[0]} {weights[1]} bias {weights[2]}")

if __name__ == "__main__":
    main()