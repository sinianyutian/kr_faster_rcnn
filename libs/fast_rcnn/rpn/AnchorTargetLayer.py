# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# based on py-faster-rcnn
# --------------------------------------------------------

from libs.fast_rcnn.config import *
from keras.engine import Layer
from libs.fast_rcnn.rpn.anchorutils import generate_anchors
import numpy as np
from libs.utils.cython_bbox import bbox_overlaps
from libs.fast_rcnn.bbox_transform import *
import tensorflow as tf
from keras import backend as Kb
from keras.layers import Input, Reshape


class AnchorTargetLayer(Layer):
    
    def __init__(self, name, **kwargs):
        print ('AnchorTargetLayer: __init__')
        anchor_scales = [int(s) for s in ANCHOR_SCALES.split(",")]
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = int(ANCHOR_FEAT_STRIDE)
        self._name = name
        #TRAIN_DEBUG = False
        
        if TRAIN_DEBUG:
            print '[DEBUG] anchor_scales: {}'.format(anchor_scales)
            print '[DEBUG] feat_stride: {}'.format(self._feat_stride)
            print '[DEBUG] anchors:'
            print self._anchors
            print '[DEBUG] anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            print '[DEBUG] num_anchors: {}'.format(self._num_anchors)
            self._counts = 1e-14 #cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0
            
        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0        
        super(AnchorTargetLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        print ('\nAnchorTargetLayer: build')
        
        #print 'input_shape: ',input_shape        
        self.built = True
        super(AnchorTargetLayer, self).build(input_shape)


    def call(self, x, mask=None):
        print ('\nAnchorTargetLayer: call')
        
        rpn_cls_score = x[0]
        gt_boxes = x[1][0]
        im_info = x[2]
        data = x[3]

        rpn_cls_score = Kb.get_value(rpn_cls_score)
        gt_boxes = Kb.get_value(gt_boxes)
        im_info = Kb.get_value(im_info)
        data = Kb.get_value(data)

        # remove zeros rows
        #print ("bef gt_boxes: {}".format(gt_boxes))
        gt_boxes = gt_boxes[~(gt_boxes == 0).all(1)]
        #print ("aft gt_boxes: {}".format(gt_boxes))

        height = int(rpn_cls_score.shape[1])
        width  = int(rpn_cls_score.shape[2])
        
        if data.shape[0] != 1:
            print '[ERROR] input_data[0]: {}'.format(data.shape[0])
            raise("[ERROR] Only single item batches are supported")
        
        if TRAIN_DEBUG:
            print '[DEBUG] im_size: ({}, {})'.format(data.shape[1], data.shape[2])
            print '[DEBUG] scale: {}'.format(data.shape[3])
            print '[DEBUG] height, width: ({}, {})'.format(height, width)
            print '[DEBUG] rpn: gt_boxes.shape', gt_boxes.shape
            #print '[DEBUG] rpn: gt_boxes', gt_boxes
            
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
                            
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)
        
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < int(data.shape[2]) + self._allowed_border) &  # width
            (all_anchors[:, 3] < int(data.shape[1]) + self._allowed_border)    # height
        )[0]

        if TRAIN_DEBUG:
            print '[DEBUG] total_anchors', total_anchors
            print '[DEBUG] inds_inside', len(inds_inside)
            
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if TRAIN_DEBUG:
            print '[DEBUG] anchors.shape', anchors.shape
            
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        
        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        
        # fg label: above threshold IOU
        labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1
        
        # subsample positive labels if we have too many
        num_fg = int(float(RPN_FG_FRACTION) * int(RPN_BATCHSIZE))
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            
        # Parametization of 4 coordinates based on gt bbox (*t_x,*t_y,*t_h,*t_w)
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
        
        # weights to down anchors which do not match gt
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array([float(w) 
                                                for w in TRAIN_BBOX_INSIDE_WEIGHTS.split(",")])
        
        # weights to normalize loss (y_true,y_pred)
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((RPN_POSITIVE_WEIGHT > 0) &
                    (RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights
        
        if TRAIN_DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print '[DEBUG] means:'
            print means
            print '[DEBUG] stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
        
        if TRAIN_DEBUG:
            print '[DEBUG] rpn: max max_overlap', np.max(max_overlaps)
            print '[DEBUG] rpn: num_positive', np.sum(labels == 1)
            print '[DEBUG] rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print '[DEBUG] rpn: num_positive avg', self._fg_sum / self._count
            print '[DEBUG] rpn: num_negative avg', self._bg_sum / self._count
            
        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        
        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        
        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        
        if TRAIN_DEBUG:
            print '[DEBUG] output: labels.shape ', labels.shape
            print '[DEBUG] output: bbox_targets.shape ', bbox_targets.shape
            print '[DEBUG] output: bbox_inside_weights.shape ', bbox_inside_weights.shape
            print '[DEBUG] output: bbox_outside_weights.shape ', bbox_outside_weights.shape
            
        self.output1 = labels.shape
        self.output2 = bbox_targets.shape
        self.output3 = bbox_inside_weights.shape
        self.output4 = bbox_outside_weights.shape
                
        #labels = Input(tensor=labels, shape=labels.get_shape())
        #ph_labels = Kb.placeholder(shape=labels.shape, dtype=labels.dtype)
        """
        labels = Input(tensor=labels, shape=labels.get_shape())
        if Kb.is_keras_tensor(labels):
            print 'tensor: yes'
        else :
            print 'tensor: no'
        """
        
        #"""
        rpn_labels = Kb.variable(value=labels, name="rpn_labels")
        rpn_bbox_targets = Kb.variable(value=bbox_targets)
        rpn_bbox_inside_weights = Kb.variable(value=bbox_inside_weights)
        rpn_bbox_outside_weights = Kb.variable(value=bbox_outside_weights)
        #"""
        
        """
        rpn_labels = Input(shape=labels.shape)
        rpn_bbox_targets = Input(shape=bbox_targets.shape)
        rpn_bbox_inside_weights = Input(shape=bbox_inside_weights.shape)
        rpn_bbox_outside_weights = Input(shape=bbox_outside_weights.shape)
        """
        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]
        #return labels
    
    """
    def compute_output_shape(self, input_shape):
        return [self.output1,self.output2,self.output3,self.output4]
    #"""
    
    #"""
    def get_output_shape_for(self, input_shape):
        
        return [self.output1, self.output2, self.output3, self.output4]
    #"""
    
    def compute_mask(self, input_tensors, input_masks):
        return [None, None, None, None]
        
    def get_config(self):
        config = {'anchors': self._anchors,
                  'num_anchors': self._num_anchors
                  }
        return config
        

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
