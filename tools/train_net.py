#!/usr/bin/env python

# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# --------------------------------------------------------
# usage:
# python train_net.py --network vgg16 --gpu 0 
# --pretrained models/tf/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
# --output output --dataset voc_2007 --epochs 10 --batch-size 1

from __future__ import print_function
import argparse, os
from libs.datasets.PascalVoc import PascalVoc
from libs.fast_rcnn.config import *
from libs.fast_rcnn.train import get_training_roidb
from libs.fast_rcnn.models.VGG16 import VGG16
from libs.fast_rcnn.objectives import softmaxWithLoss, smoothL1Loss
from libs.datasets.IMDBGenerator import IMDBGenerator
from keras.layers import Input
from keras.optimizers import SGD
import os.path as osp
from keras import backend as K
import numpy as np
import tensorflow as tf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True, help="name of network to build")
ap.add_argument("-g", "--gpu", type=int, default=0, help="id of GPU device")
ap.add_argument("-p", "--pretrained", type=str, default='default', 
                help="pretrained model to use for initialization")
ap.add_argument("-o", "--output", required=True, help="output directory")
ap.add_argument("-d", "--dataset", required=True, help="dataset name to use")
ap.add_argument("-r", "--rand", default=42, help="randomize value (seed)")
ap.add_argument("-e", "--epochs", type=int, default=20, help="# of epochs")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="size of mini-batches passed to network")
ap.add_argument("-v", "--verbose", type=int, default=1,
                help="verbosity level")
args = vars(ap.parse_args())

print('[INFO] Called with args:')
print(args)

# prepare & extract roidb entries
dataset_DIR = osp.join(ROOT_DIR, "data", args["dataset"])
print ('[INFO] loading dataset {} for training...'.format(args["dataset"]))
imdb = PascalVoc(args["dataset"], "trainval", 
                 dataset_DIR, "config/classes.lst")

print ('[INFO] set proposal method {:s} for training'.format(PROPOSAL_METHOD))
imdb.set_proposal_method(PROPOSAL_METHOD)
roidb = get_training_roidb(imdb)
# find min & max rois len
imdb.find_rois_lens()
print ('[INFO] min_rois_len per image: {}'.format(imdb.min_rois_len))
print ('[INFO] max_rois_len per image: {}'.format(imdb.max_rois_len))
print ('[INFO] {:d} roidb entries'.format(len(roidb)))
print ('[INFO] output will be saved to `{:s}`'.format(args["output"]))

# filtering roidb
num = len(roidb)
roidb = imdb.filter_roidb(roidb)
num_after = len(roidb)
print ('[INFO] Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                   num, num_after))
imgRows, imgCols, numChannels = (224,224,3)
numClasses = imdb.num_classes
print("[INFO] prepare imdb generators...")
imdb_datagen = IMDBGenerator(imdb,roidb)
imdb_generator = imdb_datagen.flow_from_pascalvoc(
    # target directory of images ordered by sub-folders
    osp.join(ROOT_DIR, "data", args["dataset"],"JPEGImages"),
    target_size=(imgRows, imgCols),  # resize images to 227x227 | 224x224
    batch_size=1, # # only 1 image per batch allowed
   )

# build & load training
print("[INFO] build & load training...")
"""
input_gt_boxes = Input(shape=(5,), name='gt_boxes') # GT boxes (x1, y1, x2, y2, label)
input_im_info = Input(shape=(3,), name='im_info') # im_height, im_width
input_data = Input(shape=(imgRows, imgCols, numChannels), name='data', dtype='float32') # h,w,c
#"""

"""
# initial data (required to set session)
input_gt_boxes = np.array([[5,112,110,115,1],
                    [115,115,97,105,2]])
input_gt_boxes = K.variable(value=input_gt_boxes)
input_im_info = K.variable(value=np.array([224,224,1]))
input_data = K.variable(value=np.random.random((1,224,224,3)))
#"""

#"""
#input_gt_boxes = np.array([[5,112,110,115,1],[115,115,97,105,2]])
input_gt_boxes = np.zeros((1,imdb.max_rois_len,5))
input_gt_boxes[0][0] = [5,112,110,115,1]
input_gt_boxes = K.variable(value=input_gt_boxes)
input_gt_boxes = Input(tensor=input_gt_boxes, shape=input_gt_boxes.get_shape())

input_im_info = np.array([[224,224,1]])
input_im_info = K.variable(value=input_im_info)
input_im_info = Input(tensor=input_im_info, shape=input_im_info.get_shape())

input_data = np.random.random((1,224,224,3))
input_data = K.variable(value=input_data)
input_data = Input(tensor=input_data, shape=input_data.get_shape())
#"""

model = VGG16(input_tensor=[input_gt_boxes, input_im_info, input_data], \
                            weights_notop_path=args["pretrained"], \
                            numClasses=numClasses)

print("[INFO] compiling model...")
sgd = SGD(lr=LEARNING_RATE, decay=WEIGHT_DECAY, momentum=MOMENTUM, nesterov=True)
losses = [softmaxWithLoss(name="rpn_loss_cls"), # rpn_loss_cls
          smoothL1Loss(name="rpn_loss_bbox"), # rpn_loss_bbox
          softmaxWithLoss(name="loss_cls"), # loss_cls
          smoothL1Loss(name="loss_bbox") # loss_bbox
          ]  
metrics = {'dense_class_{}_loss'.format(numClasses): 'accuracy'}
model.compile(loss=losses, optimizer=sgd, metrics=metrics)

print("[INFO] starting training...")
model.fit_generator(
        imdb_generator,
        #samples_per_epoch=len(roidb), # only 1 image per batch allowed
        samples_per_epoch=5002, # only 1 image per batch allowed
        nb_epoch=args["epochs"],
        verbose=args["verbose"])

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
out_model_pn = osp.join(args["output"],args["dataset"]+"_"+args["epochs"]+".hdf5")
model.save_weights(args["model"]) 
