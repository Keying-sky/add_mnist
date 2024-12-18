import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """ Load and preprocess the combined_mnist dataset. """

    data = np.load(path.DATA_DIR / 'combined_mnist.npz')
    
    x_train = data['x_train'].astype('float32')
    y_train = data['y_train']
    x_validation = data['x_validation'].astype('float32')
    y_validation = data['y_validation']
    x_test = data['x_test'].astype('float32')
    y_test = data['y_test']
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def combined_classifier(x_train, y_train, x_test, y_test, n_samples):
    """ The linear classifier for combined images."""
    
    indices = np.random.choice(len(x_train), n_samples, replace=False)
    x_train_subset = x_train[indices]
    y_train_subset = y_train[indices]
    
    # flatten the data
    x_train_flat = x_train_subset.reshape(n_samples, -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    
    # standardise the flatten data
    scaler = StandardScaler()
    x_train_flat = scaler.fit_transform(x_train_flat)
    x_test_flat = scaler.transform(x_test_flat)
    
    # train model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000, C=1.0, solver='lbfgs')
    model.fit(x_train_flat, y_train_subset)
    
    # predict
    y_pred = model.predict(x_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def separate_classifier(n_samples):
    """ The linear classifier for separate images."""
    
    # load the mnist data and normalise it
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    x_train_mnist = x_train_mnist.astype('float32') / 255.0
    x_test_mnist = x_test_mnist.astype('float32') / 255.0
    
    # flatten the data
    x_train_flat = x_train_mnist.reshape(-1, 28*28)
    x_test_flat = x_test_mnist.reshape(-1, 28*28)
    
    indices = np.random.choice(len(x_train_flat), n_samples, replace=False)
    x_train_subset = x_train_flat[indices]
    y_train_subset = y_train_mnist[indices]
    
    # train model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000, C=1.0, solver='lbfgs')
    model.fit(x_train_subset, y_train_subset)
    
    # load the test set
    combined_data = np.load('data/combined_mnist.npz')
    x_test_combined = combined_data['x_test']
    y_test_combined = combined_data['y_test']
    
    # split the left and right parts
    test_left = x_test_combined[:, :, :28]  
    test_right = x_test_combined[:, :, 28:] 
    
    test_left_flat = test_left.reshape(len(test_left), -1)
    test_right_flat = test_right.reshape(len(test_right), -1)
    
    # predict
    pred_left = model.predict(test_left_flat)
    pred_right = model.predict(test_right_flat)
    pred_sum = pred_left + pred_right
    
    return accuracy_score(y_test_combined, pred_sum)
