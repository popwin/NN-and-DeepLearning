#-*-coding:utf-8-*-


import numpy as np
import theano
from theano import tensor as T

class SoftMax():
    """作为输出层
    """
    def __init__(self,input,n_in,n_out):
        
        W = np.asarray((n_in,n_out),dtype=theano.config.floatX)
        self.W = theano.shared(value=W,name='W',borrow=True)
        
        b = np.asarray((n_out,),dtype=theano.config.floatX)
        self.b = theano.shared(value=b,name='b',borrow=True)
        
        self.output = T.nnet.softmax(T.dot(input,self.W)+self.b)#得到tensor变量
        self.y_pred = T.argmax(self.output,axis=1)
        self.params = [self.W,self.b]
        self.input = input
    
    def cost_function(self,y):
        """用负号变为取最小值，且用log处理为和
           y为tensor变量，一个label向量（非one-hot）
        """
        Y = T.log(self.output)[T.arange(y.shape[0]),y]#取出label对应的估计值，1{y==j}log
        return -T.mean(Y)
    
    def errors(self,y):
        return T.mean(T.neq(self.y_pred,y))#neq返回0或者1，当y_pred ==y 返回1，否则0



