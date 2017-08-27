# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# based on py-faster-rcnn
# --------------------------------------------------------

import os, PIL, scipy
from libs.fast_rcnn.config import *
#from libs.utils.cython_bbox import bbox_overlaps
import numpy as np

class IMDB(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}
        self._dataset_DIR = None
        
    @property
    def dataset_DIR(self):
        return self._dataset_DIR
        
    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)
    
    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index
        
    @property
    def cache_path(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        return CACHE_DIR
    
    def _get_widths(self):
        return [PIL.Image.open(osp.join(self.dataset_DIR, "JPEGImages",im_n+".jpg")).size[0]
              for im_n in self.image_index]
        
    def default_roidb(self):
        raise NotImplementedError
       
    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method
        
    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb
    
    @roidb.setter
    def roidb(self, val):
        self._roidb = val
    
    @property
    def num_images(self):
        return len(self.image_index)
    
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            # if roidb is None, load first via roidb_handler
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            # new xmin & xmax of horizontal-flipped image
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            #assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     "seg_areas": self.roidb[i]['seg_areas'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
    
    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        """
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes' : boxes,
                'gt_classes' : np.zeros((num_boxes,), dtype=np.int32),
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : np.zeros((num_boxes,), dtype=np.float32),
            })
        #"""
        return roidb
