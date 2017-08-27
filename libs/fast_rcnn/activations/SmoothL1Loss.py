# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# --------------------------------------------------------

from keras.engine.topology import Layer
from libs.fast_rcnn.objectives import smoothL1Loss
import tensorflow as tf
from libs.fast_rcnn.config import *
from keras import backend as Kb

class SmoothL1Loss(Layer):
    
    def __init__(self, name, sigma, **kwargs):
        print ('\nSmoothL1Loss: __init__')
        self._name = name
        self._sigma = sigma
        self._phase = 0
        
        super(SmoothL1Loss, self).__init__(**kwargs)
        
    def build(self, input_shape):
        print ('\nSmoothL1Loss: build')
              
        self.built = True
        super(SmoothL1Loss, self).build(input_shape)
        
    def call(self, x, mask=None):
        bbox_pred = x[0]
        bbox_targets = x[1]
        bbox_inside_weights = x[2]
        bbox_outside_weights = x[3]
        
        if TRAIN_DEBUG:
            print ("bbox_pred.shape: {}".format(bbox_pred.get_shape()))
            print ("bbox_targets.shape: {}".format(bbox_targets.get_shape()))
            print ("bbox_inside_weights.shape: {}".format(bbox_inside_weights.get_shape()))
            print ("bbox_outside_weights.shape: {}".format(bbox_outside_weights.get_shape()))
        
        sigma2 = self._sigma * self._sigma
        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
        
        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
    
        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
        
        smoothL1Loss = None
        if self._sigma == 3.0: # smoothL1Loss 1
            smoothL1Loss = tf.reduce_mean(tf.reduce_sum(outside_mul, reduction_indices=[1, 2, 3]))
        elif self._sigma == 1.0: # smoothL1Loss 2
            smoothL1Loss = tf.reduce_mean(tf.reduce_sum(outside_mul, reduction_indices=[1]))

        print ("SmoothL1LossLayer, {}: {}".format(self._name, Kb.get_value(smoothL1Loss)))
        self.output1 = smoothL1Loss.shape
        if TRAIN_DEBUG:
            #print ("smoothL1Loss.shape: {}".format(smoothL1Loss.shape))
            pass
        
        return smoothL1Loss
        
    #"""
    def get_output_shape_for(self, input_shape):
        
        return [None]
    #"""
    
    def compute_mask(self, input_tensors, input_masks):
        return [None]

    def get_config(self):
        config = {'activation': 'SmoothL1Loss'}
        return dict(list(config.items()))
