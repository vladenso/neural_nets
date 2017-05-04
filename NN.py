
# coding: utf-8

# In[1]:

import numpy as np
import sklearn as sk
import scipy.io as sp
from random import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[2]:

Data = sp.loadmat("letters_data.mat")


# In[3]:

X = Data['train_x']
Y = Data['train_y']
Test = Data['test_x']


# In[4]:

l = list(range(len(Y)))
shuffle(l)
x, y = [], [] 
for i in l:
    x.append(X[i])
    y.append(Y[i])
X = x
Y = y


# In[5]:

X.extend(list(Test))
data = StandardScaler(X, with_mean=True)
X, Test = X[:len(Y)], X[len(Y):]


# In[ ]:




# In[6]:

Y_hot_enc = []
for y in Y:
    yhe = np.array([0]*26)
    yhe[y[0]-1] = 1
    Y_hot_enc.append(yhe)
Y = Y_hot_enc


# In[7]:

o = [1]*len(X)
X = np.column_stack((X, o))
o = [1]*len(Test)
Test = np.column_stack((Test, o))


# In[8]:

def sigmoid(g):
    return 1/(1 + np.exp(-g))


# In[ ]:




# In[9]:

class Neural_Network(object):
    def __init__(self):
        self.input_size = 784 + 1
        self.hidden_size = 200 + 1 
        self.output_size = 26
        
        self.V = np.random.randn(self.hidden_size, self.input_size - 1)
        self.V = np.column_stack((self.V, [1]*(self.hidden_size)))
        self.W = np.random.randn(self.output_size, self.hidden_size)
        
    def forward(self, X):
        self.h = np.tanh(np.dot(self.V, X))
        self.z = sigmoid(np.dot(self.W, self.h))
        return self.z
    
    def cost(self, x, y):
        z = self.forward(x)
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
        
    def cost_fn_prime(self, y):
        d = self.z - y
        dw = np.outer(d, self.h)
        
        dl_dh = np.dot(np.transpose(self.W), d)
        t = np.tanh(self.h)
        tp = (1 - t * t)
        dh_dv = np.transpose(np.outer(x, tp))
        for i in range(len(dl_dh)):
            dh_dv[i] *= dl_dh[i]
        return dh_dv, dw
    
    def train(self, images, labels, l1, l2):
        loss = []
        for n in range(1):
            l1 = l1/2
            l2 = l2/2
            for i in range(len(images)):
                #i = j  #np.random.randint(len(images))
                x, y = images[i], labels[i]
                z = self.forward(x)
                V, W = self.cost_fn_prime(y)
                self.V -= l1* V
                self.W -= l2* W
                #loss.append(self.cost(x, z))
            print(n)
            
        return loss
            
    def predict(self, image):
        return self.forward(image)


# In[ ]:
print("Building NN")

NN = Neural_Network()


# In[ ]:
print("Training NN")
n = 1# len(Y)//10
l = NN.train(X[:n], Y[:n], .1, .1)

print("Trained NN")
# In[ ]:

plt.plot(range(n), l)
plt.show()


# In[ ]:

error()


# In[ ]:

def error():
    e = 0
    pr = len(Y) -n -1
    for i in range(pr):
        p = NN.predict(X[n+i])
        if numericle(p) != numericle(Y[n+i]):
            e += 1
    return e/pr


# In[ ]:

def numericle(l):
    max(l, key=lambda k: l[k])
    for i in range(len(l)):
        if l[i] > .9:
            return i


# In[ ]:

L = [.01, .05, .1, .25, .4]
for l in L:
    NN = Neural_Network()
    NN.train(X[:n], Y[:n], l/2, l)
    print(error())


# In[ ]:

v, w = n.cost_fn_prime(X[0], Y[0])


# In[ ]:

W, V = n.W, n.V


# In[ ]:

scal = .25


# In[ ]:

n.W = n.W - scal*w
n.V = n.V - scal*v
cost2 = n.cost(X[0], Y[0])


# In[ ]:

n.W = W + scal*w
n.V = V + scal*v
cost3 = n.cost(X[0], Y[0])


# In[ ]:

print(cost1, cost2, cost3)


# In[ ]:

l = np.random.randn(5, 3)
print(l)
l = np.column_stack((l, [1]*5))
print(l)


# In[ ]:



