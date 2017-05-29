# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# based on py-faster-rcnn
# --------------------------------------------------------
import numpy as np
from keras import backend as Kb
import tensorflow as tf
from libs.fast_rcnn.config import *
from keras.engine import Layer
from libs.fast_rcnn.rpn.anchorutils import generate_anchors
from libs.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from libs.fast_rcnn.nms_wrapper import nms
from keras.layers import Input

class ProposalLayer(Layer):
    
    def __init__(self, name, **kwargs):
        print ('\nProposalLayer: __init__')
        anchor_scales = [int(s) for s in ANCHOR_SCALES.split(",")]
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = int(ANCHOR_FEAT_STRIDE)
        self._name = name
        #TRAIN_DEBUG = False
        
        if TRAIN_DEBUG:
            print '[DEBUG] feat_stride: {}'.format(self._feat_stride)
            print '[DEBUG] anchors:'
            print self._anchors
            
        super(ProposalLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        print ('\nProposalLayer: build')
        
        self.built = True
        super(ProposalLayer, self).build(input_shape)
        
    def call(self, x, mask=None):
        print ('\nProposalLayer: call')
        
        rpn_cls_prob_reshape = x[0]
        rpn_bbox_pred = x[1]
        im_info = x[2]

        rpn_cls_prob_reshape = Kb.get_value(rpn_cls_prob_reshape)
        rpn_bbox_pred = Kb.get_value(rpn_bbox_pred)
        im_info = Kb.get_value(im_info)
        im_info = im_info[0]

        rpn_cls_prob_reshape = rpn_cls_prob_reshape.reshape((1,18,14,14))
        #print "rpn_cls_prob_reshape.shape: ", rpn_cls_prob_reshape.shape
        #print "rpn_bbox_pred.shape: ", rpn_bbox_pred.shape
        #print "im_info.shape: ", im_info.shape

        '''
        if self._phase == 0:
            rpn_cls_prob_reshape = np.random.random((1,18,14,14))
            rpn_bbox_pred = np.random.random((1,14,14,36))
            im_info = np.array([224,224,1])
            self._phase = 1
        else:
            #"""
            # create a tf session to convert tensor
            # & variable to numpy array
            if TRAIN_DEBUG:
                print "creating a tf session..."
            sess = tf.InteractiveSession()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
                    
            rpn_cls_prob_reshape = rpn_cls_prob_reshape.eval()
            rpn_bbox_pred = rpn_bbox_pred.eval()
            im_info = im_info.eval()           
            sess.close()
            #"""
        '''

        if rpn_cls_prob_reshape.shape[0] != 1:
            print '[ERROR] input_data[0]: {}'.format(rpn_cls_prob_reshape.shape[0])
            raise("[ERROR] Only single item batches are supported")
        
        pre_nms_topN, post_nms_topN, nms_thresh, min_size = (None, None, None, None)
        
        if TRAIN_DEBUG:
            print '[DEBUG] RUN_PHASE: {}'.format(RUN_PHASE)
            
        if RUN_PHASE == "train":
            pre_nms_topN  = TRAIN_RPN_PRE_NMS_TOP_N
            post_nms_topN = TRAIN_RPN_POST_NMS_TOP_N
            nms_thresh    = TRAIN_RPN_NMS_THRESH
            min_size      = TRAIN_RPN_MIN_SIZE
        elif RUN_PHASE == "test":
            pre_nms_topN  = TEST_RPN_PRE_NMS_TOP_N
            post_nms_topN = TEST_RPN_POST_NMS_TOP_N
            nms_thresh    = TEST_RPN_NMS_THRESH
            min_size      = TEST_RPN_MIN_SIZE
        else:
            raise("[ERROR] unknown execution phase")        
        
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = rpn_cls_prob_reshape[:, self._num_anchors:, :, :]
        bbox_deltas = rpn_bbox_pred
        
        if TRAIN_DEBUG:
            print '[DEBUG] im_size: ({}, {})'.format(im_info[0], im_info[1])
            print '[DEBUG] scale: {}'.format(im_info[2])
            
        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        height, width = int(height), int(width)
        
        if TRAIN_DEBUG:
            print '[DEBUG] scores.shape: {}'.format(scores.shape)
            
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
                            
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        #print "anchors.shape: ",anchors.shape
        #print "bbox_deltas.shape: ", bbox_deltas.shape
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        
        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        
        if TRAIN_DEBUG:
            print '[DEBUG] output: blob.shape ', blob.shape
            print '[DEBUG] output: scores.shape ', scores.shape
            
        self.output1 = blob.shape
        self.output2 = scores.shape
        
        #"""
        rpn_rois = Kb.variable(value=blob, name="rpn_rois")
        rpn_scores = Kb.variable(value=scores, name="rpn_scores")
        #"""

        return [rpn_rois, rpn_scores]
    
    #"""
    def get_output_shape_for(self, input_shape):        
        return [self.output1, self.output2]
    #"""
    
    def compute_mask(self, input_tensors, input_masks):
        return [None, None]
        
        
    def get_config(self):
        print ('\nProposalLayer: get_config')
        config = {'anchors': self._anchors,
                  'num_anchors': self._num_anchors
                  }
        return config
    
def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep 
