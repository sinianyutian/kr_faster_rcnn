# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# --------------------------------------------------------

from keras.engine.topology import Layer
from libs.fast_rcnn.objectives import softmaxWithLoss
import tensorflow as tf
from libs.fast_rcnn.config import *
from keras import backend as Kb

class SoftmaxWithLoss(Layer):
    
    def __init__(self, name, **kwargs):
        print ('\nSoftmaxWithLoss: __init__')
        self._name = name
        self._phase = 0
        
        super(SoftmaxWithLoss, self).__init__(**kwargs)
        
    def build(self, input_shape):
        print ('\nSoftmaxWithLoss: build')
              
        self.built = True
        super(SoftmaxWithLoss, self).build(input_shape)
        
    def call(self, x, mask=None):
        logits = x[0]
        labels = x[1]
        
        if TRAIN_DEBUG:
            print ("logits.shape: {}".format(logits.get_shape()))
            print ("labels.shape: {}".format(labels.get_shape()))
         
    
        cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                   labels=labels)
        softmaxWithLoss = tf.reduce_mean(cross_entropy_with_logits)
        print ("softmaxWithLossLayer, {}: {}".format(self._name, Kb.get_value(softmaxWithLoss)))
        
        self.output1 = softmaxWithLoss.shape
        
        if TRAIN_DEBUG:
            #print ("softmaxWithLoss.shape: {}".format(softmaxWithLoss.shape))
            pass
        
        return softmaxWithLoss
    
    #"""
    def get_output_shape_for(self, input_shape):
        
        return [self.output1]
    #"""
    
    def compute_mask(self, input_tensors, input_masks):
        return [None]

    def get_config(self):
        config = {'activation': 'SoftmaxWithLoss'}
        return dict(list(config.items()))
    
