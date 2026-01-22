import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

random.seed(1)

w1 = [[random.random(), random.random()],
      [random.random(), random.random()]]
b1 = [random.random(), random.random()]

w2 = [random.random(), random.random()]
b2 = random.random()

learning_rate = 0.5
epochs = 5000

for _ in range(epochs):
    for x, y in data:
        h1 = sigmoid(x[0] * w1[0][0] + x[1] * w1[1][0] + b1[0])
        h2 = sigmoid(x[0] * w1[0][1] + x[1] * w1[1][1] + b1[1])

        output = sigmoid(h1 * w2[0] + h2 * w2[1] + b2)

        error = y - output
        d_output = error * sigmoid_derivative(output)

        w2[0] += learning_rate * d_output * h1
        w2[1] += learning_rate * d_output * h2
        b2 += learning_rate * d_output

        d_h1 = d_output * w2[0] * sigmoid_derivative(h1)
        d_h2 = d_output * w2[1] * sigmoid_derivative(h2)

        w1[0][0] += learning_rate * d_h1 * x[0]
        w1[1][0] += learning_rate * d_h1 * x[1]
        b1[0] += learning_rate * d_h1

        w1[0][1] += learning_rate * d_h2 * x[0]
        w1[1][1] += learning_rate * d_h2 * x[1]
        b1[1] += learning_rate * d_h2

print("Testing the trained network")
for x, y in data:
    h1 = sigmoid(x[0] * w1[0][0] + x[1] * w1[1][0] + b1[0])
    h2 = sigmoid(x[0] * w1[0][1] + x[1] * w1[1][1] + b1[1])
    output = sigmoid(h1 * w2[0] + h2 * w2[1] + b2)
    print(x, "->", round(output), "(Expected:", y, ")")
