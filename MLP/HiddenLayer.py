#-*-coding:utf-8

import numpy as np
import theano
from theano import tensor as T

class HiddenLayer():
    """
    """
    def __init__(self,rng,input,n_in,n_out,
            W=None,b=None,activate=T.tanh):
        """input:符号变量，dmatrix，n_example*n_in , n_in 数据维度，每          一行为一个sample
            W: n_in*n_out dimention,初始化时，根据不同的activate函数
              而不同，Bengio提出|sqrt(6./(fan_in+fan_out))|,fan_in
              输入层units个数，类比fan_out
              sigmoid用——4*|sqrt(6./(fan_in+fan_out))|
            rng:随机数生成器
        """   
        self.input = input
        
        if W is None:
            W_val = np.asarray(rng.uniform(
                                low=-np.sqrt(6./(n_in+n_out)),
                                high=-np.sqrt(6./(n_in+n_out)),
                                size=(n_in,n_out)                               
                                 ),
                               dtype=theano.config.floatX)#W 初始化
            W = theano.shared(value=W_val,name='W',borrow=True)#转化为shared变量已适应GPU运算
            
        if b is None:
            b_val = np.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=b_val,name='b',borrow=True)
        
        self.W = W
        self.b = b
        
        z = T.dot(input,self.W)+self.b
        self.output = (z if activate is None else activate(z))


if __name__ == '__main__':







