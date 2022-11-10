import numpy #sum da errores
from numpy import array, exp, maximum, zeros, arange, argmax
from numpy.random import rand, shuffle
from pandas import read_csv
from matplotlib.pyplot import gray, imshow, show 

data = array(read_csv('train.csv'))
m, n = data.shape
shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[: 1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1 : n]
X_dev = X_dev / 255

data_train = data[1000 : m].T
Y_train = data_train[0]
X_train = data_train[1 : n]
X_train = X_train / 255
_,m_train = X_train.shape

def init_params():
    return tuple(rand(10, n) - 1 / 2 for n in (784, 1, 10, 1))

def ReLU(Z):
    return maximum(Z, 0)

def softmax(Z):
    e = exp(Z)
    
    return e / sum(e)
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = zeros((Y.size, Y.max() + 1))
    one_hot_Y[arange(Y.size), Y], one_hot_Y = 1, one_hot_Y.T

    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    dZ2 = A2 - one_hot(Y)
    dW2, db2, dZ1 = 1 / m * dZ2.dot(A1.T), 1 / m * numpy.sum(dZ2),\
                    W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1, db1 = 1 / m * dZ1.dot(X.T), 1 / m * numpy.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    return W1 - alpha * dW1, b1 - alpha * db1, W2 - alpha * dW2, b2 - alpha * db2

def get_predictions(A2):
    return argmax(A2, 0)

def get_accuracy(predictions, Y):
    return numpy.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    wsbs = init_params()
    W1, b1, W2, b2 = wsbs

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(*wsbs, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        wsbs = update_params(*wsbs, dW1, db1, dW2, db2, alpha)
        W1, b1, W2, b2 = wsbs

        if i % 10 == 0:          
            predictions = get_predictions(A2)
            print("Iteration: ", i, '\n', get_accuracy(predictions, Y))

    return (*wsbs,)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)

    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]

    print("Prediction: ", prediction, "\nLabel: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    gray()
    imshow(current_image, interpolation='nearest')
    show()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1 / 10, 500)
tuple(test_prediction(n, W1, b1, W2, b2) for n in range(4))
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(dev_predictions, Y_dev, '\n', get_accuracy(dev_predictions, Y_dev))
