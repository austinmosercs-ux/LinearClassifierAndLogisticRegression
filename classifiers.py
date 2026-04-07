# Homework 5 - Linear Classifier and Logistic Regression
# CSCI 405/CIS 605 Artificial Intelligence
#
# Implements and compares two binary classification approaches trained
# with online (per-sample) gradient descent on a small 2-feature dataset:
#
#   Part 1 - Linear Classifier with Hard Threshold:
#     Uses a step activation (output = 1 if net >= 0.5 else 0).
#     Weight update rule:  w_i += lr * (y - y_hat) * x_i
#
#   Part 2 - Logistic Regression:
#     Uses the sigmoid activation so the output is a probability in (0, 1).
#     Weight update rule (generalized delta):  w_i += lr * delta * x_i
#       where delta = (y - sigma(net)) * sigma(net) * (1 - sigma(net))
#
# Both parts share the same learning rate, convergence threshold, and
# initial weights so the results are directly comparable.

import math

# Training data rows: [X1, X2, Y]
# Y=1 denotes the positive class, Y=0 the negative class.
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

# Four points used to evaluate the trained decision boundary,
# including an extreme point (20, 100) to test extrapolation.
test_samples = [(1, 8), (2, 1.5), (9, 5), (20, 100)]

# Hyperparameters shared by both classifiers.
learning_rate = 0.001   # step size for each weight update
MAX_ITER = 10000000     # hard cap on training epochs to prevent infinite loops
THRESHOLD = 1e-10       # convergence criterion: max weight change per epoch

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
    """Logistic sigmoid: maps any real number to the open interval (0, 1)."""
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

    # Same L-inf convergence check as Part 1.
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
