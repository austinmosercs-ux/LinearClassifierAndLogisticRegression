# Homework 5 - Linear Classifier and Logistic Regression
# CSCI 405/CIS 605 Artificial Intelligence

import math

# training data [X1, X2, Y]
data = [
    [1,   6,   1],
    [1.5, 7,   1],
    [2,   2,   0],
    [2.5, 5.5, 1],
    [3,   4,   0],
    [3.5, 7,   1],
    [5,   5,   0],
    [5.5, 6,   1],
    [6,   7,   0],
    [6.5, 8.5, 0]
]

# test samples
test_samples = [(1, 8), (2, 1.5), (9, 5), (20, 100)]

learning_rate = 0.001
MAX_ITER = 10000000
THRESHOLD = 1e-10

# -------------------------------------------------------
# Part 1: Linear Classifier with Hard Threshold
# -------------------------------------------------------

print("=" * 55)
print("Part 1: Linear Classifier with Hard Threshold")
print("=" * 55)

# initial weights
w0 = 0.1
w1 = 0.2
w2 = 0.3

print(f"Initial weights: w0 = {w0}, w1 = {w1}, w2 = {w2}")
print(f"Learning rate: {learning_rate}")
print()

converged = False
for i in range(MAX_ITER):
    prev_w0 = w0
    prev_w1 = w1
    prev_w2 = w2

    for sample in data:
        x1, x2, y = sample

        net = w0 + w1 * x1 + w2 * x2

        # hard threshold - output 1 if net >= 0.5 else 0
        if net >= 0.5:
            y_hat = 1
        else:
            y_hat = 0

        error = y - y_hat

        # update rule
        w0 = w0 + learning_rate * error
        w1 = w1 + learning_rate * error * x1
        w2 = w2 + learning_rate * error * x2

    # check convergence - if the change in weights is tiny enough we stop
    diff = max(abs(w0 - prev_w0), abs(w1 - prev_w1), abs(w2 - prev_w2))
    if diff < THRESHOLD:
        print(f"Converged after {i + 1} iterations")
        converged = True
        break

if not converged:
    print(f"Did not converge, stopped at max iterations ({MAX_ITER})")

print(f"Final weights: w0 = {w0:.8f}, w1 = {w1:.8f}, w2 = {w2:.8f}")

# accuracy on training data
correct = 0
print("\nTraining set predictions:")
for sample in data:
    x1, x2, y = sample
    net = w0 + w1 * x1 + w2 * x2
    if net >= 0.5:
        y_hat = 1
    else:
        y_hat = 0
    status = "correct" if y_hat == y else "wrong"
    print(f"  X1={x1}, X2={x2}  ->  predicted={y_hat}, actual={y}  ({status})")
    if y_hat == y:
        correct += 1

accuracy = correct / len(data) * 100
print(f"\nTraining Accuracy: {correct}/{len(data)} = {accuracy:.1f}%")

print("\nTest samples:")
for x1, x2 in test_samples:
    net = w0 + w1 * x1 + w2 * x2
    if net >= 0.5:
        y_hat = 1
    else:
        y_hat = 0
    print(f"  (X1={x1}, X2={x2})  ->  predicted class = {y_hat}")


# -------------------------------------------------------
# Part 2: Logistic Regression Classifier
# -------------------------------------------------------

print()
print("=" * 55)
print("Part 2: Logistic Regression Classifier")
print("=" * 55)

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

# reset initial weights
w0 = 0.1
w1 = 0.2
w2 = 0.3

print(f"Initial weights: w0 = {w0}, w1 = {w1}, w2 = {w2}")
print(f"Learning rate: {learning_rate}")
print()

converged = False
for i in range(MAX_ITER):
    prev_w0 = w0
    prev_w1 = w1
    prev_w2 = w2

    for sample in data:
        x1, x2, y = sample

        net = w0 + w1 * x1 + w2 * x2
        output = sigmoid(net)

        error = y - output

        # generalized delta rule with sigmoid derivative
        delta = error * output * (1 - output)

        w0 = w0 + learning_rate * delta
        w1 = w1 + learning_rate * delta * x1
        w2 = w2 + learning_rate * delta * x2

    diff = max(abs(w0 - prev_w0), abs(w1 - prev_w1), abs(w2 - prev_w2))
    if diff < THRESHOLD:
        print(f"Converged after {i + 1} iterations")
        converged = True
        break

if not converged:
    print(f"Did not converge, stopped at max iterations ({MAX_ITER})")

print(f"Final weights: w0 = {w0:.8f}, w1 = {w1:.8f}, w2 = {w2:.8f}")

# accuracy on training data
correct = 0
print("\nTraining set predictions:")
for sample in data:
    x1, x2, y = sample
    net = w0 + w1 * x1 + w2 * x2
    output = sigmoid(net)
    if output >= 0.5:
        y_hat = 1
    else:
        y_hat = 0
    status = "correct" if y_hat == y else "wrong"
    print(f"  X1={x1}, X2={x2}  ->  output={output:.4f}, predicted={y_hat}, actual={y}  ({status})")
    if y_hat == y:
        correct += 1

accuracy = correct / len(data) * 100
print(f"\nTraining Accuracy: {correct}/{len(data)} = {accuracy:.1f}%")

print("\nTest samples:")
for x1, x2 in test_samples:
    net = w0 + w1 * x1 + w2 * x2
    output = sigmoid(net)
    if output >= 0.5:
        y_hat = 1
    else:
        y_hat = 0
    print(f"  (X1={x1}, X2={x2})  ->  output={output:.4f}, predicted class = {y_hat}")
