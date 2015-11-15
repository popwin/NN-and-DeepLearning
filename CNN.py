
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

import theano


# In[3]:

from theano import tensor as T


# In[4]:

from theano.tensor.nnet import conv


# In[5]:

from theano.tensor.signal import downsample


# In[21]:

import sys
sys.path.append(r"xxx")


# In[22]:

from MLP import HiddenLayer


# In[23]:

from MLP import SoftMax


# In[24]:

from MLP import load_data


# In[25]:

class CNN():
    """这里每一层都采用：下一层每个map由上一层所有map组成
    """
    def __init__(self,rng,input,filter_shape,
                 image_shape,pool_size=(2,2)):
        """input :输入符号变量，4唯向量，[mini_batch_size,input层的feature map数，image height，image width]
           filter_shape:(输出层feature map数量，输入的feature map数量，filter高，filter宽)
           image_shape：(batch_size，输入feature map数量，image高，image宽)
           filter_shape相当于构建卷积层的那张连接表，只不过这里后一层的每个map和前一层都连接
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        #fan_in/fan_out同样用于初始化W
        #fan_in的设计原则是：m-1层map数量*filter height*filter width
        #fan_out的设计原则是：m层map数量*filter height*filter width/poolsize
        fan_in = np.prod(filter_shape[1:])
        fan_out = np.prod(filter_shape[0]*np.prod(filter_shape[2:])
                          /np.prod(pool_size))
        
        W_bound = np.sqrt(6./(fan_in+fan_out))
        #卷积层的W数目和连接表有关，这里为“全连”，那么共有
        #m层map数*m-1层map数 个W，
        self.W = theano.shared(np.asarray(
                                          rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                                          dtype=theano.config.floatX),
                               borrow=True)
        b_val = np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b = theano.shared(value=b_val,borrow=True)
        
        #建立卷积层,conv2d需要两个参数，一个是4d的tensor变量input
        #一个是4d的参数W——[m层map数，m-1层map数，filter heght，filter width]
        conv_out = conv.conv2d(input=input,filters=self.W,
                               filter_shape=filter_shape,image_shape=image_shape)
        
        #建立pool采样层
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=pool_size,
                                            ignore_border=True)
        #这里要将b进行维度转化，转化成一个4d向量，1*m层map数*1*1
        self.output = T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))
        
        self,params = [self.W,self.b]
        self.input =input
        
        


# In[26]:

def main(path,alpha=0.01,epochs=200,
         nkerns=[20,50],batch_size=500):
    """
    """
    #####load data####
    rng = np.random.RandomState(23455)
    datasets = load_data(path)
    
    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    
    index = T.lscalar()
    
    x = T.matrix('x')#是一个sample矩阵
    y = T.ivector('y')#x对应的label，是一个向量
    
    #####build model####
    print "building model"
    input_layer = x.reshape((batch_size,1,28,28))
    layer1 = CNN(rng,#第一个卷积层
                 input=input_layer,
                 filter_shape=(nkerns[0],1,5,5),
                 image_shape=(batch_size,1,28,28)
                 )
    layer2 = CNN(rng,#第二个卷积层
                 input=layer1.output,
                 filter_shape=(nkerns[1],nkerns[0],5,5),
                 image_shape=(batch_size,nkerns[0],12,12)
                 )
    
    #将layer2的输出flatten为2维向量，（batch_size,nkerns[1]*4*4）
    layer3_input = layer2.output.flatten(2)
    layer3 = HiddenLayer(rng,
                         input=layer3_input,
                         n_in=nkerns[1]*4*4,
                         n_out=500)
    
    layer4 = SoftMax(input=layer3.output,
                     n_in=500,
                     n_out=10)
    
    cost = layer4.cost_function(y)
    
    params = layer4.params+layer3.params+layer2.params+layer1.params
    grads = T.grad(cost,params)
    updates = [(param,param-alpha*grad) 
               for param,grad in zip(params,grads)
               ]
    train_model = theano.function([index],
                                  cost,
                                  updates=updates,
                                  givens={x:train_set_x[index*batch_size:(index+1)*batch_size],
                                           y:train_set_y[index*batch_size:(index+1)*batch_size]}
                                   )
    
    valid_model = theano.functionn([index],
                                   layer4.errros(y),
                                   givens={x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                           y:valid_set_y[index*batch_size:(index+1)*batch_size]}
                                   )


    
    ######eairly stop paraments####
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
            iter = (epoch-1)*n_train_batches+batch_index
            if iter%100 == 0:
                print "training @iter=%d"%iter
            train_cost = train_model(batch_index)
            
            
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
    
    
    
    


# In[53]:

main("/home/pop/Downloads/mnist.pkl.gz")


# In[28]:

rng = np.random.RandomState(23455)
input=np.ones((3,4))
x=HiddenLayer(rng,input,n_in=4,n_out=6)


# In[ ]:

#np.prod(x) 将x中所有元素相乘
#

