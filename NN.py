
# coding: utf-8

# In[1]:

import numpy as np
import random
import mnist_loader


# In[31]:

class NN():
    """规定theta为Nout*Nin矩阵
       每一个sample或者unit为一个列向量
       偏置bias作为独立输入，加入到除input外的每层units中
    """
    def __init__(self,struct):
        self.layers = len(struct)
        self.struct = struct#一个包含各层多少个节点的列表
        #self.bias是一个包含隔层bias的列表，【0】是第一层
        self.bias = [np.random.randn(units,1) for units in struct[1:]]
        #self.theta是一个包含隔层theta的列表，每个theta以后一层units各数为行
        self.theta = [np.random.randn(m,n) 
                      for m,n in zip(struct[1:],struct[:-1])]
                
    
    def backprop(self,x,y):
        """
        """
        grad_b = [np.zeros(b.shape) for b in self.bias]
        grad_theta = [np.zeros(theta.shape) for theta in self.theta]
        a = x
        zs = []
        activate = [x]
        for b,theta in zip(self.bias,self.theta):
            z = np.dot(theta,a)+b
            zs.append(z)
            a = self.sigmoid(z)
            activate.append(a)
        
        #back
        delta = (activate[-1]-y)#the output error of cross-entropy
        grad_b[-1] = delta #the grad b in output 
        grad_theta[-1] = np.dot(delta,activate[-2].transpose())#grad theta in output layer
    
        for l in xrange(2,self.layers):
            sp = activate[-l]*(1-activate[-l])
            delta = np.dot(self.theta[-l+1].transpose(),delta)*sp
            grad_b[-l] = delta
            grad_theta[-l] = np.dot(delta,activate[-l-1].transpose())
        
        return (grad_b,grad_theta)
    
    def mini_batch(self,batch,alpha):
        """batch is a list of tuples(x,y)
        """
        Grad_B = [np.zeros(b.shape) for b in self.bias]
        Grad_Theta = [np.zeros(w.shape) for w in self.theta]
        for x,y in batch:
            grad_b,grad_theta = self.backprop(x,y)
            Grad_B = [item1+item2 
                      for item1,item2 in zip(Grad_B,grad_b)]    
            Grad_Theta = [item1+item2 
                          for item1,item2 in zip(Grad_Theta,grad_theta)]
        
        self.theta = [theta-alpha*Dtheta
                      for theta,Dtheta in zip(self.theta,Grad_Theta)]
        self.bias = [b-alpha*Db 
                    for b,Db in zip(self.bias,Grad_B)]
     
    def SGD(self,training_data,
            epochs,batch_size,alpha,test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)    
        
        for i in xrange(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size]
                       for k in xrange(0,n,batch_size)]
            for batch in batches:
                self.mini_batch(batch,alpha)
        
        if test_data:
            print "Epoch {0}:{1}/{2}".format(i,self.evaluate(test_data),n_test)
        else:
            print"Epoch {0} complete".format(i)
    
    
    def evaluate(self,test_data):
        """
        """
        result = [(np.argmax(self.OutPut(x)),y) for (x,y) in test_data]
        return sum((int(x == y)) for (x,y) in result)
    
    def OutPut(self,a):
        for theta,b in zip(self.theta,self.bias):
            a = self.sigmoid(np.dot(theta,a)+b)
        return a                                   
            
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
        
        


# In[32]:

import mnist_loader


# In[33]:

training_data,validation_data,test_data=mnist_loader.load_data_wrapper()


# In[34]:

net=NN([784,30,10])


# In[ ]:

net.SGD(training_data,30,10,3.0,test_data=test_data)


# In[ ]:



