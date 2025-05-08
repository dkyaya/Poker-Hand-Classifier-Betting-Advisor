import numpy as np
import pandas as pd

# Load and process poker hand dataset
data = pd.read_csv('/Users/yayaj/Desktop/Java - VSC/AI/FinalProj/Training Sets/poker-hand-training-true.data', header=None)
data = np.array(data)

m, n = data.shape  # m = number of rows, n = 11 (10 features + 1 label)
np.random.shuffle(data)

# Separate dev and train sets
data_dev = data[:1000].T
Y_dev = data_dev[-1]  # Last column is the label
X_dev = data_dev[:-1]  # First 10 columns are features
X_dev = X_dev / np.max(X_dev)  # Normalize features

data_train = data[1000:].T
Y_train = data_train[-1]
X_train = data_train[:-1]
X_train = X_train / np.max(X_train)
_, m_train = X_train.shape

# Neural network parameters and functions
def init_params():
    W1 = np.random.rand(10, 10) - 0.5  # 10 hidden units, 10 input features
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5  # 10 output classes
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z - np.max(Z, axis=0))  # stability
    return A / A.sum(axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2 @ A1.T
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * ReLU_deriv(Z1)
    dW1 = 1 / m_train * dZ1 @ X.T
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(f"Iteration {i} Accuracy: {get_accuracy(predictions, Y):.4f}")
    return W1, b1, W2, b2

# Train model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=200)

# Evaluation
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(index, W1, b1, W2, b2):
    current_input = X_dev[:, index, None]
    prediction = make_predictions(current_input, W1, b1, W2, b2)
    label = Y_dev[index]
    print(f"Prediction: {prediction[0]} | Label: {label}")
