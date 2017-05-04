
# coding: utf-8

# In[1]:

import numpy as np
import sklearn as sk
import scipy.io as sp
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# In[2]:
# Load Data
Data = sp.loadmat("letters_data.mat")


# In[3]:

X = Data['train_x']
Y = Data['train_y']
Test = Data['test_x']


# In[4]:
# Shuffle the data
l = list(range(len(Y)))
shuffle(l)
x, y = [], [] 
for i in l:
    x.append(X[i])
    y.append(Y[i])
X = x
Y = y


# In[5]:
# Scaling and normilizing training and test data
ss = StandardScaler()
X.extend(Test)

X = ss.fit_transform(X)
X = normalize(X)
X, Test = X[:len(Y)], X[len(Y):]


# In[6]:
# One hot encoding test outputs
Y_hot_enc = []
for y in Y:
    yhe = np.array([0]*26)
    yhe[y[0]-1] = 1
    Y_hot_enc.append(yhe)
Y = Y_hot_enc


# In[7]:
# Bias term for input
o = [1]*len(X)
X = np.column_stack((X, o))
o = [1]*len(Test)
Test = np.column_stack((Test, o))


# In[8]:

def sigmoid(g):
    return 1/(1 + np.exp(-g))


# In[9]:

class Neural_Network(object):
    def __init__(self):
        self.input_size = 784 + 1
        self.hidden_size = 200  
        self.output_size = 26
        self.n = 1
        
        self.V = np.random.randn(self.hidden_size, self.input_size) #*.1
        self.W = np.random.randn(self.output_size, self.hidden_size + 1) #*.1
    
    def forward(self, x):
        self.h = np.tanh(np.dot(self.V, x))

        self.h = np.concatenate((self.h, np.array([1])))
        self.z = sigmoid(np.dot(self.W, self.h))
        return self.z
    
        
    def cost_fn_prime(self, x, y):
        d = self.forward(x) - y
        dw = np.outer(d, self.h) 
        
        dl_dh = np.dot(np.transpose(self.W), d)
        dv = np.outer((dl_dh[:200] * (1 - self.h[:200] * self.h[:200])), x)
        return dv, dw
    
    def train(self, images, labels, l1, l2, epoch, ep, l_reg, runs):
        loss, err = [], []
        for n in range(epoch):
            for j in range(runs):
                i = np.random.randint(len(images))
                dV, dW = self.cost_fn_prime(images[i], labels[i])
                self.V -= l1* dV
                self.W -= l2* (dW + l_reg * self.W)
            loss.append(self.cost(self.z, labels[i]))
            l1 = l1*ep
            l2 = l2*ep
            err.append(error())
            print(error()) 
        return loss, err
            
    def predict(self, image):
        return self.forward(image)
    
    def cost(self, z, y):
        s = 0
        for i in range(26):
            l, r = 0, 0 
            if z[i] == 1:
                zs = .99999
            elif z[i] == 0:
                zs = .00001
            else: 
                zs = z[i]
            if y[i] == 0:
                r = np.log(1 - zs)
            elif y[i] == 1:
                l = np.log(zs)
            else:
                r = (1 - y[i])* np.log(1 - zs)
                l = y[i] * np.log(zs)
            s -= l + r
        return s    
    


# In[10]:

def numericle(l):
    return np.argmax(l)


# In[11]:

def error():
    e = 0
    pr = len(Y) -n -1
    for i in range(pr):
        p = NN.predict(X[n+i])
        if numericle(p) != numericle(Y[n+i]):
            e += 1
    return e/pr


# In[12]:

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
           'u', 'v', 'w', 'x', 'y', 'z']
def visualize():
    i = 0
    c, inc = 0, 0
    print("Correct")
    while c <5:
        p = NN.predict(X[n+i])
        if numericle(p) == numericle(Y[n+i]):
            print("Cassified as: ", letters[numericle(p)])
            x = np.reshape(X[n+i][:-1], (28,28))
            plt.imshow(x)
            plt.show()
            c +=1
        i+=1
    i = 0
    print("Incorrect")
    while inc <5:
        p = NN.predict(X[n+i])
        if numericle(p) != numericle(Y[n+i]):
            print("Cassified as: ", letters[numericle(p)])
            print("Correct label: ", letters[numericle(Y[n+i])])
            x = np.reshape(X[n+i][:-1], (28,28))
            plt.imshow(x)
            plt.show()
            inc +=1
        i+=1


# In[15]:

print(X[1])


# In[13]:

NN = Neural_Network()


# In[474]:

l = .95
ep = 50
n = len(Y)//10*8
r = 10000
l, e = NN.train(X[:n], Y[:n],  0.006, 0.01, ep, l,  .0000, r)


# In[485]:

plt.plot(range(len(l)), l)
plt.show()
plt.plot(range(len(e[:100])), e[:100])
plt.show()


# In[476]:

1-error()


# In[496]:

visualize()


# Kaggle

# In[341]:

predictions = []
for t in Test:
    p = numericle(NN.predict(t)) + 1
    predictions.append(p)


# In[342]:

import csv
with open('kaggle.csv', 'w', newline='') as csvfile: 
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL) 
    writer.writerow(["Id", "Category"])
    i=1
    for l in predictions: 
        writer.writerow([i, l]) 
        i+=1


# Used for hyperparameter tunning.

# In[76]:

n = len(Y)//10*8
#L1 = [0, .000001, .00001, .0001, .001, .01, .1]#
L1 = [.001, .005, .01, .05]
#L1 = [5, 4, 3, 2]
L2 = [.01, .1, .25, .5, .75]
#L2 = [1]#[.5, .7, .9]
for l1 in L1:
    for l2 in L2:
        NN = Neural_Network()
        NN.W = np.copy(Wc) 
        NN.V = np.copy(Vc) 
        NN.train(X[:n], Y[:n], l1, l2, 5, .95, 0, 1000)
        print(error(), l1, l2)


# In[ ]:

print(cost1, cost2, cost3)


# In[192]:

n = 5000
#L1 = [0, .00001, .0001, .001, .01]#
L1 = [0.025, .05, .075]
#L1 = [2, 3, 4, 5]
L2 = [0.0025, .004, .006]
#L2 = [.5, .6, .7, .8, .9]
for l1 in L1:
    for l2 in L2:
        NN = Neural_Network()
        #NN.train(X[:n], Y[:n],  .01, .0025, l1, l2)#, 4, .9)
        NN.train(X[:n], Y[:n], l1, l2, 1, .8, .005)
        print(error(), l1, l2)

