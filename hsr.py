import numpy as np
import pandas as pd

data = pd.read_csv('/Users/yayaj/Desktop/Java - VSC/AI/FinalProj/Training Sets/poker-hand-training-true.data')
data = np.array(data)# converts to numpy array

m, n = data.shape # save number of rows and columns
np.random.shuffle(data) # shuffle before splitting into dev and training sets

#variables to use for testing after the training
data_dev = data[0:1000].T #transpose a thousand images
Y_dev = data_dev[-1] #should now be the labels - digits
X_dev = data_dev[:-1] 
X_dev = X_dev / np.max(X_dev)  # Normalize features


#same thing with training data 1000 through 4K
data_train = data[1000:].T #using these rows for training
Y_train = data_train[-1]
X_train = data_train[:-1]
X_train = X_train / np.max(X_train) 
_,m_train = X_train.shape

#define random weights and biases
def init_params():
   W1 = np.random.rand(10, 10) - 0.5
   b1 = np.random.rand(10, 1) - 0.5
   W2 = np.random.rand(10, 10) - 0.5
   b2 = np.random.rand(10, 1) - 0.5
   return W1, b1, W2, b2

def ReLU(Z):
   return np.maximum(Z, 0)

def softmax(Z):
   A = np.exp(Z - np.max(Z, axis=0)) 
   return A / A.sum(axis=0)

def forward_prop(W1, b1, W2, b2, X):
   Z1 = W1.dot(X) + b1
   A1 = ReLU(Z1)
   Z2 = W2.dot(A1) + b2
   A2 = softmax(Z2)
   return Z1, A1, Z2, A2

def ReLU_deriv(Z):
   return Z > 0

def one_hot(Y):
   one_hot_Y = np.zeros((Y.size, Y.max() + 1))
   one_hot_Y[np.arange(Y.size), Y] = 1
   one_hot_Y = one_hot_Y.T
   return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
   one_hot_Y = one_hot(Y)
   dZ2 = A2 - one_hot_Y
   dW2 = 1 / m * dZ2.dot(A1.T)
   db2 = 1 / m * np.sum(dZ2)
   dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
   dW1 = 1 / m * dZ1.dot(X.T)
   db1 = 1 / m * np.sum(dZ1)
   return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
   W1 = W1 - alpha * dW1
   b1 = b1 - alpha * db1   
   W2 = W2 - alpha * dW2 
   b2 = b2 - alpha * db2   
   return W1, b1, W2, b2

def get_predictions(A2):
   return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
   print(predictions, Y)
   return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
   W1, b1, W2, b2 = init_params()
   for i in range(iterations):
       Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
       dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
       W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
       if i % 10 == 0:
           print("Iteration: ", i)
           predictions = get_predictions(A2)
           print(get_accuracy(predictions, Y))
   return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1000)

def make_predictions(X, W1, b1, W2, b2):
   _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
   predictions = get_predictions(A2)
   return predictions

def test_prediction(index, W1, b1, W2, b2):
   current_image = X_dev[:, index, None]
   prediction = make_predictions(X_dev[:, index, None], W1, b1, W2, b2)
   label = Y_dev[index]
   print("Prediction: ", prediction)
   print("Label: ", label)
