# Perceptron Learning in Python

import random

InputVectorSize = 2
WeightSize = InputVectorSize + 1  # weights + bias
LEARNING_RATE = 0.1
ITERATIONS = 10

weights = [0.0] * WeightSize

def initialize():
    global weights
    # Initialize the weights with random values
    weights = [random.random() for _ in range(WeightSize)]

def feedforward(InputVectorSizes):
    sum = 0.0
    # Calculate InputVectorSizes * weights
    for i in range(InputVectorSize):
        sum += weights[i] * InputVectorSizes[i]
    # Add in the bias
    sum += weights[InputVectorSize]
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
            desired_output = InputVectorSizes[0] and InputVectorSizes[1] # To train to OR vs and simply change and to or in this line and vice versa
            output = feedforward(InputVectorSizes)
            error = desired_output - output

            print(f"{InputVectorSizes[0]} or {InputVectorSizes[1]} = {output} ({desired_output})")

            # Update weights
            weights[0] += LEARNING_RATE * error * InputVectorSizes[0]
            weights[1] += LEARNING_RATE * error * InputVectorSizes[1]
            weights[2] += LEARNING_RATE * error

            iteration_error += error ** 2

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