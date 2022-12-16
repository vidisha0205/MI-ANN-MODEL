from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import math
df = pd.read_csv('WineQT.csv')
columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
df.drop(['Id'],axis='columns',inplace=True)
df.dropna(subset=['quality'],inplace=True)
df['best quality'] = [1 if x > 6 else 0 for x in df.quality]
target = df['best quality']
features = df.drop(['quality'], axis=1)
labels = np.unique(features['best quality'].values)
idx_to_labels = { k:v for k,v in enumerate(labels) }
labels_to_idx = { v:k for k,v in enumerate(labels) }
labels = features.replace(labels_to_idx)['best quality'].values
df = features.drop(columns=['best quality'])
# one hot encoding
labels = np.eye(len(idx_to_labels))[labels]
df = (df-df.mean())/df.std()

# replace NaN with Standard Deviation
df = df.fillna(df.std())
np.random.seed(1291)
features = df.values
print(features.shape)
print("BEGIN TRAINING")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
gamma = {}
gamma["ndims"] = features.shape[1]
gamma["nclasses"] = len(idx_to_labels.values())
def generate_weights(gamma):
    '''
        Generate Weights and use Xavier Initiation
    '''
    scale = 1/max(1., (2+2)/2.)
    limit = math.sqrt(3.0 * scale)

    gamma['w0'] = np.random.uniform(-limit, limit, size=(gamma['ndims'], gamma['ndims']))
    gamma['w1'] = np.random.uniform(-limit, limit, size=(gamma['ndims'], gamma['nclasses']))
    
    return gamma
gamma = generate_weights(gamma)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dsigmoid(x):
    return x * (1. - x)
def loss(y, y_hat):
    '''
        Addition of all Squared Mean Errors
    '''
    return np.sum(np.mean(np.square(np.subtract(y, y_hat)), axis=0))
def forward(X, gamma):
    '''
        Forward Propagation
    '''
    l0 = X
    l1 = sigmoid(np.dot(l0, gamma['w0']))
    l2 = sigmoid(np.dot(l1, gamma['w1']))
    
    return l0, l1, l2

def backward(y, theta, gamma, lr):
    '''
        Backward Propagation
    '''
    l0, l1, l2 = theta
    
    l2_error = y - l2
    l2_delta = l2_error * dsigmoid(l2)
    l1_error = l2_delta.dot(gamma['w1'].T)
    l1_delta = l1_error * dsigmoid(l1)
    # update using SGD
    gamma['w0'] += lr * l0.T.dot(l1_delta)
    gamma['w1'] += lr * l1.T.dot(l2_delta)
    
    return gamma

def train(X, y, gamma, iterations=300, lr=0.0605):
    '''
        Function to Train Dataset
    '''
    errors = []
    for i in range(iterations):
        # forward propagation
        theta = forward(X, gamma)
        
        e = loss(theta[-1], y)
        if(i % 4 == 0):
            print('I:{0:4d}, --  Mean Error:{1:1.4f}'.format(i, np.mean(e)))
        errors.append(e)
        # backward propagation
        gamma = backward(y, theta, gamma, lr)
    return gamma, errors

gamma, errors = train(X_train, y_train, gamma)

def accuracy_measure(y_true, y_pred):
    acc=0
    for i in range(len(y_true)):
        if(y_true[i][0]==y_pred[i][0] and y_true[i][1]==y_pred[i][1]):
            acc=acc+1
    return acc/len(y_true)

def accuracy(y, gamma):
    '''
    Function to calculate accuracy
    '''
    acc_y=[]
    for x in X_test:
        g = np.argmax(forward(x.reshape(1, 11), gamma)[-1])
        g = np.eye(gamma["nclasses"])[g]
        acc_y.append(g)
    acc_y = np.array(acc_y)
    return accuracy_measure(y,acc_y)

print('Accuracy:{0:3d}%'.format(int(accuracy(y_test, gamma) * 100)))

