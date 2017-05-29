# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# based on py-faster-rcnn
# --------------------------------------------------------
from keras.engine import Layer
import numpy as np
import numpy.random as npr
import tensorflow as tf
from libs.fast_rcnn.config import *
from libs.fast_rcnn.bbox_transform import *
from libs.utils.cython_bbox import bbox_overlaps
from keras import backend as Kb
from keras.layers import Input

class ProposalTargetLayer(Layer):
    
    def __init__(self, name, numClasses, **kwargs):
        print ('\nProposalTargetLayer: __init__')
        
        self._name = name
        self._numClasses = numClasses
        self._count = 0
        self._fg_num = 0
        self._bg_num = 0
        #TRAIN_DEBUG = False
        
        super(ProposalTargetLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        print ('\nProposalTargetLayer: build')
        
        self.built = True
        super(ProposalTargetLayer, self).build(input_shape)
        
        
    def call(self, x, mask=None):
        print ('\nProposalTargetLayer: call')
        
        rpn_rois = x[0]
        gt_boxes = x[1][0]

        rpn_rois = Kb.get_value(rpn_rois)
        gt_boxes = Kb.get_value(gt_boxes)

        # remove zeros rows
        gt_boxes = gt_boxes[~(gt_boxes == 0).all(1)]

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (rpn_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'
                
        num_images = 1
        rois_per_image = ROI_BATCH_SIZE / num_images 
        fg_rois_per_image = np.round(RPN_FG_FRACTION * rois_per_image)
        
        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._numClasses)

        if TRAIN_DEBUG:
            print '[DEBUG] num fg: {}'.format((labels > 0).sum())
            print '[DEBUG] num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            ratio = 0
            if self._bg_num > 0:
                ratio = float(self._fg_num) / float(self._bg_num)
            print '[DEBUG] num fg avg: {}'.format(self._fg_num / self._count)
            print '[DEBUG] num bg avg: {}'.format(self._bg_num / self._count)
            print '[DEBUG] ratio: {:.3f}'.format(ratio)
            
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        
        if TRAIN_DEBUG:
            print '[DEBUG] output: rois.shape ', rois.shape
            print '[DEBUG] output: labels.shape ', labels.shape
            print '[DEBUG] output: bbox_targets.shape ', bbox_targets.shape
            print '[DEBUG] output: bbox_inside_weights.shape ', bbox_inside_weights.shape
            print '[DEBUG] output: bbox_outside_weights.shape ', bbox_outside_weights.shape
            
        self.output1 = rois.shape
        self.output2 = labels.shape
        self.output3 = bbox_targets.shape
        self.output4 = bbox_inside_weights.shape
        self.output5 = bbox_outside_weights.shape
        
        all_labels = np.zeros((self._numClasses, labels.shape[0]), dtype=gt_boxes.dtype)
        print '[DEBUG] labels: ', labels
        labels = [int(v) for v in labels]
        all_labels[labels] = 1
        #print '[DEBUG] output: labels ', all_labels
        
        #"""
        rois = Kb.variable(value=rois)
        labels = Kb.variable(value=all_labels)
        bbox_targets = Kb.variable(value=bbox_targets)
        bbox_inside_weights = Kb.variable(value=bbox_inside_weights)
        bbox_outside_weights = Kb.variable(value=bbox_outside_weights)
        #"""
        
        """
        rois = Input(shape=rois.shape)
        labels = Input(shape=labels.shape)
        bbox_targets = Input(shape=bbox_targets.shape)
        bbox_inside_weights = Input(shape=bbox_inside_weights.shape)
        bbox_outside_weights = Input(shape=bbox_outside_weights.shape)
        #"""
        return [rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
        
    #"""
    def get_output_shape_for(self, input_shape):        
        return [self.output1, self.output2, self.output3, self.output4, self.output5]
    #"""
    
    def compute_mask(self, input_tensors, input_masks):
        return [None, None, None, None, None]
        
    def get_config(self):
        config = {'anchors': self._anchors,
                  'num_anchors': self._num_anchors
                  }
        return config
        
        

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        '''
        print ("cls: {}").format(cls)
        print ("ind: {}").format(ind)
        print ("start: {}").format(start)
        print ("end: {}").format(end)
        print bbox_target_data[ind, 1:]
        print bbox_targets[ind, start:end]
        '''
        try:
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = TRAIN_BBOX_INSIDE_WEIGHTS
        except:
            print ('Catched => ValueError: could not broadcast input array from shape (4) into shape (0)')
       
    return bbox_targets, bbox_inside_weights        

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    """
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    """
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)        
        

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights        
        
