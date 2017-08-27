# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# --------------------------------------------------------

import os
from keras.preprocessing.image import Iterator, load_img, img_to_array
from keras import backend as K
import numpy as np
from libs.datasets.utils import load_im_data

class PascalVocIterator(Iterator):
    
    def __init__(self, directory, imdb_generator,
                 target_size=(224, 224), color_mode='rgb',
                 dim_ordering='default',
                 batch_size=1, seed=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.imdb_generator = imdb_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        
        self.white_list_formats = ['png', 'jpg', 'jpeg', 'bmp']
        
        self.nb_class = self.imdb_generator.imdb.num_classes
        self.class_indices = self.imdb_generator.imdb.class_to_ind
        self.nb_im_sample = self.imdb_generator.imdb.num_images
        self.nb_roi_sample = len(self.imdb_generator.roidb)
        print('[INFO] Found {} imdb belonging to dataset {}'.\
              format(self.nb_im_sample ,self.imdb_generator._imdb.name))               
        print('[INFO] Found {} roidb belonging to {} classes.'.\
              format(self.nb_roi_sample,self.nb_class))
        super(PascalVocIterator, self).__init__(self.nb_im_sample, batch_size, None, seed)
        # initialize our flow_index
        #self.index_generator = self.flow_index(self.nb_im_sample, batch_size, shuffle, seed)
        
    def flow_index(self, nb_im, batch_size=1, shuffle=False, seed=None):
        raise NotImplementedError
    
    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        if current_batch_size != 1:
            raise("ERROR: batch size image required is 1, found {}".format(current_batch_size))
        
        batch_data = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        #print 'batch_data.shape: ', batch_data.shape
        batch_im_info = np.zeros((1,3))
        # roi_entry (rois) related to current image (in batch_data)
        roi_entry = self.imdb_generator._roidb[current_index]
        # gt_boxes = (x,y,h,w,label)
        batch_gt_boxes = np.zeros((self.imdb_generator.imdb.max_rois_len,5))
        gt_boxes = None
        
        # build batch of image data
        #print 'index_array: ', index_array
        for i, j in enumerate(index_array):
            im_fn = self.imdb_generator._imdb._image_index[j]
            im_pn = None
            for ext in self.white_list_formats:
                im_pn = os.path.join(self.directory, im_fn+"."+ext)
                if os.path.exists(im_pn):
                    break
            assert im_pn != None
                
            # load & perform data transformation
            im, gt_boxes = load_im_data(im_pn, roi_entry, 
                 target_size=self.target_size, dim_ordering=self.dim_ordering)
            x = img_to_array(im, dim_ordering=self.dim_ordering)            
            batch_data[i] = x
        
        # build batch of image info
        for i, j in enumerate(index_array):
            #print "self.target_size, ",self.target_size
            batch_im_info[:,[0,1]] = np.array(self.target_size)
            batch_im_info[:,-1:] = 1
        
        # build batch of gt_boxes
        for i, gt_bbox in enumerate(gt_boxes):
            batch_gt_boxes[i] = gt_bbox
        #print 'batch_gt_boxes.shape: ', batch_gt_boxes.shape

        X = [np.copy(np.array([batch_gt_boxes])), np.copy(batch_im_info), np.copy(batch_data)]
        # Y (y_true) Will be calculate inside the network
        #Y = [np.array([[0,0]]),np.array([[0,0]]),np.array([[0,0]]),np.array([[0,0]])]
        Y = [np.random.random((1,21)), np.random.random((1,21)),np.random.random((1,21)),np.random.random((1,21))]

        return X, Y
    
