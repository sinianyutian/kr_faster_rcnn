# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# --------------------------------------------------------

"""
    VGG16 model for Keras, adapted for faster-rcnn
"""

import os
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.layers.core import Reshape, Activation, Dropout
from keras.engine.topology import get_source_inputs
from libs.fast_rcnn.rpn.AnchorTargetLayer import AnchorTargetLayer
from libs.fast_rcnn.rpn.ProposalLayer import ProposalLayer
from libs.fast_rcnn.rpn.ProposalTargetLayer import ProposalTargetLayer
from libs.fast_rcnn.roi_data_layer.RoiPooling import RoiPooling
from libs.fast_rcnn.activations.SmoothL1Loss import SmoothL1Loss
from libs.fast_rcnn.activations.SoftmaxWithLoss import SoftmaxWithLoss
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape

def VGG16_BASE(inputs=None, weights_notop_path=None):
            
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(inputs)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
            
    # Create model.
    model = Model(inputs, x, name='vgg16_faster_rcnn_end2end')
    
    if os.path.exists(weights_notop_path):
        model.load_weights(weights_notop_path)
        return model
    else:
        raise ("[ERROR] unable to load weights_notop of model: {}".format(weights_notop_path))
        return None


def VGG16(input_tensor=None, weights_notop_path=None, numClasses=None):

    K.set_learning_phase(0)
    
    gt_boxes = input_tensor[0]
    im_info = input_tensor[1]
    data = input_tensor[2]

    """
    input_gt_boxes = Input(shape=(5,), name='gt_boxes') # GT boxes (x1, y1, x2, y2, label)
    input_im_info = Input(shape=(2,), name='im_info') # im_height, im_width
    input_data = Input(shape=(imgRows, imgCols, numChannels), name='data') # h,w,c    
    """
    
    """ 
        Initial VGG16 model without top layer 
    """
    print 'data.get_shape: ', data.get_shape()
    base_model = VGG16_BASE(inputs=data, weights_notop_path=weights_notop_path)
    
    """ 
        Add rpn & fast-rcnn layers to base_model 
    """    
    """ ========= RPN ========= """
    conv6_1 = ZeroPadding2D((1,1))(base_model.output)
    conv6_1 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv6_1")(conv6_1)
    print 'conv6_1.get_shape : ', conv6_1.get_shape()
    rpn_cls_score = Convolution2D(18, 1, 1, subsample=(1,1), name="conv6_2")(conv6_1) # 2(bg/fg) * 9(anchors)
    rpn_bbox_pred = Convolution2D(36, 1, 1, subsample=(1,1), name="conv6_3")(rpn_cls_score) # 4 * 9(anchors)    
    rpn_cls_score_reshape = Reshape((1,2,-1,18), name="rpn_cls_score_reshape")(rpn_cls_score) # 2 * 9(anchors)
    rpn_cls_score_reshape._keras_history = rpn_cls_score._keras_history
    #print 'rpn_cls_score_reshape.get_shape : ', rpn_cls_score_reshape.get_shape()
    #print 'rpn_cls_score_reshape._keras_history: ', rpn_cls_score_reshape._keras_history

    input_dim = [rpn_cls_score,gt_boxes,im_info,data]
    rpn_data = AnchorTargetLayer(name="rpn-data")(input_dim)
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data
    
    logits1 = tf.reshape(rpn_cls_score_reshape, shape=[-1,2])
    labels1 = K.cast(K.variable(value=rpn_labels),tf.int32)
    labels1 = tf.reshape(labels1, shape=[-1])
        
    logits1 = tf.reshape(tf.gather(logits1,tf.where(tf.not_equal(labels1,-1))),[-1,2])
    labels1 = tf.reshape(tf.gather(labels1,tf.where(tf.not_equal(labels1,-1))),[-1])   

    softmaxWithLoss1 = SoftmaxWithLoss(name="softmaxWithLoss1")([logits1,labels1])
    softmaxWithLoss1._keras_history = rpn_cls_score_reshape._keras_history
    softmaxWithLoss1._keras_shape = (1,21)
    #print 'softmaxWithLoss1._keras_history: ', softmaxWithLoss1._keras_history
    
    rpn_bbox_targets1 = tf.transpose(rpn_bbox_targets,[0,2,3,1])
    rpn_bbox_inside_weights1 = tf.transpose(rpn_bbox_inside_weights,[0,2,3,1])
    rpn_bbox_outside_weights1 = tf.transpose(rpn_bbox_outside_weights,[0,2,3,1])

    #print 'rpn_bbox_pred.get_shape : ', rpn_bbox_pred.get_shape()
    input_dim = [rpn_bbox_pred, rpn_bbox_targets1, rpn_bbox_inside_weights1, rpn_bbox_outside_weights1]
    smoothL1Loss1 = SmoothL1Loss(name="smoothL1Loss1", sigma=3.0)(input_dim)
    smoothL1Loss1._keras_history = rpn_bbox_targets._keras_history
    smoothL1Loss1._keras_shape = (1, numClasses)
    #print 'smoothL1Loss1._keras_history: ', smoothL1Loss1._keras_history
    
    """ ========= RoI Proposal ========= """
    rpn_cls_prob = Activation("softmax",name="rpn_cls_prob")(rpn_cls_score_reshape[0][0])
    rpn_cls_prob_reshape = Reshape((18,14,14), name="rpn_cls_prob_reshape")(rpn_cls_prob)
    rpn_cls_prob_reshape._keras_history = rpn_cls_score_reshape._keras_history
    #print 'rpn_cls_prob_reshape._keras_history: ', rpn_cls_prob_reshape._keras_history
    
    rpn_rois, _ = ProposalLayer(name="proposal")([rpn_cls_prob_reshape, rpn_bbox_pred, im_info])
    #print 'rpn_rois._keras_history: ', rpn_rois._keras_history
    roi_data = ProposalTargetLayer(name="roi-data", numClasses=numClasses)([rpn_rois, gt_boxes])
    rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data
       
    """ ========= RCNN ========= """
    #print "rois: ", rois
    #print "rois.shape: ", rois.shape
    #num_rois = int(rois.shape[1])
    num_rois = int(rois.get_shape()[0])
    
    pool5 = RoiPooling(pool_list=[7], num_rois=num_rois, 
                       spatial_scale=0.0625, name="roi_pool5")([base_model.output,rois])
    flatten = Reshape(target_shape=(-1,int(pool5.shape[1])*int(pool5.shape[2])))(pool5)
   
    fc6 = Dense(4096, activation='relu',name='fc6')(flatten)
    fc6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu',name='fc7')(fc6)
    fc7 = Dropout(0.5)(fc7)
    cls_score = Dense(numClasses,name='cls_score')(fc7)
    bbox_pred = Dense(numClasses*4,name='bbox_pred')(fc7)
    
    #logits2 = cls_score
    logits2 = tf.reshape(cls_score, shape=[-1,1,1])
    labels2 = K.cast(K.variable(value=labels),tf.int32)
    labels2 = tf.reshape(labels2, shape=[-1,1])
        
    softmaxWithLoss2 = SoftmaxWithLoss(name="softmaxWithLoss2")([logits2,labels2])
    softmaxWithLoss2._keras_history = rpn_cls_score_reshape._keras_history
    softmaxWithLoss2._keras_shape = softmaxWithLoss2.get_shape()
    #print 'softmaxWithLoss2._keras_history: ', cls_score._keras_history
    
    input_dim = [bbox_pred, bbox_targets,bbox_inside_weights, bbox_outside_weights]
    smoothL1Loss2 = SmoothL1Loss(name="smoothL1Loss2", sigma=1.0)(input_dim)
    smoothL1Loss2._keras_shape = (1, numClasses)
    #print 'smoothL1Loss2._keras_history: ', smoothL1Loss2._keras_history
    #print 'smoothL1Loss2._keras_shape: ', smoothL1Loss2._keras_shape

    print("=========Outputs: Y_pred============")
    print ("softmaxWithLoss1: {}".format(K.get_value(softmaxWithLoss1)))
    print ("smoothL1Loss1: {}".format(K.get_value(smoothL1Loss1)))
    print ("softmaxWithLoss2: {}".format(K.get_value(softmaxWithLoss2)))
    print ("smoothL1Loss2: {}".format(K.get_value(smoothL1Loss2)))
    final_output = [softmaxWithLoss1, smoothL1Loss1, softmaxWithLoss2, smoothL1Loss2]
        
    model = Model(input=[gt_boxes, im_info, base_model.input], output=final_output)
    
    return model
