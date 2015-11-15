#-*-coding:utf-8 -*-

import numpy as np
import theano
from theano import tensor as T
import gzip
import cPickle


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

if __name__ == '__main__':
    path = "/home/pop/Downloads/mnist.pkl.gz"
    data = load_data(path)
    trainx,trainy = data[0]
    print "hello"
    print trainx.get_value(borrow=True).shape[0]

