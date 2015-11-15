
# coding: utf-8


import theano
import numpy as np
from theano import tensor as T
import timeit
import os
import gzip
import cPickle

class HiddenLayer():
    """
    """
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activate=T.tanh):
        """input:符号变量，dmatrix，n_example*n_in , n_in 数据维度，每一行为一个sample
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



class MLP():
    """三层的preceptron
    """
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        """input:符号变量，
        """
        self.h = HiddenLayer(rng,input,n_in,n_out)#隐藏层
        self.outlayer = SoftMax(input=self.h.output,
                           n_in=n_hidden,
                           n_out=n_out)
        self.L2 = (self.h.W**2).sum()+(self.outlayer.W**2).sum()
        
        self.cost = self.outlayer.cost_function
        self.errors = self.outlayer.errors
        self.params = self.h.params.extend(self.outlayer.params)
        self.input = input

def load_data(data_path):

    with gzip.open(data_path,'rb') as f:
        train_set,valid_set,test_set = cPickle.load(f)
    
    def shared_datas(data,borrow=True):
        """data中每一个sample是一个tuple，包括两部分，一个是784*1的向量
           一个是该sample的label
        """
        data_x,data_y = data
        sharedX = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        sharedY = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
        
        return sharedX,T.cast(sharedY,'int32')
    
    train_set_x,train_set_y = shared_datas(train_set)
    valid_set_x,valid_set_y = shared_datas(valid_set)
    return [(train_set_x,train_set_y),(valid_set_x,valid_set_y)]


def main(alapha=0.01,lamda=0.001,epochs=1000,batch_size=30,n_hidden=500):
    
    #####Load data####
    path = "/home/pop/Downloads/mnist.pkl.gz"
    datasets = load_data(path)
    train_set_x,train_set_y = datasets[0]#训练集 ，共50000
    valid_set_x,valid_set_y = datasets[1]#评估集，共
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    
    
    ##############
    #Actual Model#
    ##############
    
    index = T.lscalar()#batch的序号
    x = T.matrix('x')#input data 每一个x是一个sample（28*28的image）
    y = T.ivector('y')#所有sample的label向量
    rng = np.random.RandomState(1234)
    
    mlp = MLP(rng=rng,input=x,n_in=28*28,n_hidden=n_hidden,n_out=10)
    Jtheta = (mlp.cost(y)+lamda*mlp.L2)
    grads = [T.grad(Jtheta,param) for param in mlp.params]
    updates = [(param,param-alapha*grad) 
               for param,grad in zip(mlp.params,grads)]
    
    train_model = theano.function(inputs=[index],outputs=Jtheta,
                                  updates=updates,
                                  givens={
                                  x:train_set_x[index*batch_size:(index+1)*batch_size],
                                  x:train_set_y[index*batch_size:(index+1)*batch_size]}
                                  )
    
    valid_model = theano.function(inputs=[index],outputs=mlp.errors(y),
                                  givens={
                                  x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                  x:valid_set_y[index*batch_size:(index+1)*batch_size]}
                                  )
    
    #######eairly stop paraments####
    #基本思想是：cost很长时间没有变化时，那么就停止，如果变化了，那么在
    #超出容忍范围后，以2倍迭代次数来提升容忍度，直到基本不变
    
    patience = 10000
    patience_inc = 2
    improv_thres = 0.995
    valid_fre = min(n_train_batches,patience/2)
    best_validation_loss = np.inf
    best_iter = 0
    start_time = timeit.default_timer()
    epoch = 0
    done_loop = False
    
    ########
    print "Model is training"
    
    while (epoch<epochs) and (not done_loop):
        epoch += 1
        for batch_index in xrange(n_train_batches):
            train_cost = train_model(batch_index)
            
            iter = (epoch-1)*n_train_batches+batch_index
            if (iter+1) % valid_fre == 0:
                valid_loss = [valid_model(i)
                              for i in xrange(n_valid_batches)]
                ave_valid_loss = np.mean(valid_loss)
                print "epoch%d--error%f"%(epoch,ave_valid_loss)
                
                if ave_valid_loss<best_validation_loss:
                    if ave_valid_loss<best_validation_loss*improv_thres:
                        patience = max(patience,iter*patience_inc)
                    best_validation_loss = ave_valid_loss
                    best_iter = iter
            
            if patience <= iter:
                done_loop = True
                break
    
    end_time = timeit.default_timer()
    print "Opt is completed "
    print "Best validation score is %f"%(best_validation_loss*100)
    

if __name__=='__main__':
    main()



#PS:np.zeros((3,))产生的是一行
#b是每一层的bias，同一层中，不同的样本当然用的是一样的bias
#tensor变量通过.eval()来访问其值
#T.argmax(y,axis=1)取每一行中最大的值的序号，5*2的矩阵，那么得到5个值，每一行的最大
#numpy中 matrix.sum()是矩阵中的元素全部相加

