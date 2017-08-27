# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# --------------------------------------------------------

import ConfigParser as cp
import json, os
import os.path as osp

''' 
    Config: DATASETS 
'''
MAIN_CFG = cp.RawConfigParser()
MAIN_CFG.read('config/main.cfg')

ROOT_DIR = MAIN_CFG.get("DATA", "ROOT_DIR")
CACHE_DIR = MAIN_CFG.get("DATA", "CACHE_DIR")

RUN_PHASE = MAIN_CFG.get("EXECUTION", "PHASE")
USE_GPU_NMS = MAIN_CFG.get("EXECUTION", "USE_GPU_NMS")
GPU_ID = MAIN_CFG.get("EXECUTION", "GPU_ID")

''' 
    Config: DATASETS 
'''
DS_CFG = cp.RawConfigParser()
DS_CFG.read('config/datasets.cfg')

TRAIN_FN = DS_CFG.get("DATA", "TRAIN_FN")
VAL_FN = DS_CFG.get("DATA", "VAL_FN")
TEST_FN = DS_CFG.get("DATA", "TEST_FN")


''' 
    Config: FASTER-RCNN
'''
FRCNN_CFG = cp.RawConfigParser()
FRCNN_CFG.read('config/faster_rcnn.cfg')

TRAIN_DEBUG = bool(FRCNN_CFG.get("TRAIN", "DEBUG"))
USE_FLIPPED = bool(FRCNN_CFG.get("TRAIN", "USE_FLIPPED"))
PROPOSAL_METHOD = FRCNN_CFG.get("TRAIN", "PROPOSAL_METHOD")
FG_THRESH = float(FRCNN_CFG.get("TRAIN", "FG_THRESH"))
BG_THRESH_HI = float(FRCNN_CFG.get("TRAIN", "BG_THRESH_HI"))
BG_THRESH_LO = float(FRCNN_CFG.get("TRAIN", "BG_THRESH_LO"))

RPN_POSITIVE_OVERLAP = float(FRCNN_CFG.get("TRAIN", "RPN_POSITIVE_OVERLAP"))
RPN_FG_FRACTION = float(FRCNN_CFG.get("TRAIN", "RPN_FG_FRACTION"))
RPN_BATCHSIZE = int(FRCNN_CFG.get("TRAIN", "RPN_BATCHSIZE"))
TRAIN_BBOX_INSIDE_WEIGHTS = FRCNN_CFG.get("TRAIN", "TRAIN_BBOX_INSIDE_WEIGHTS")
RPN_POSITIVE_WEIGHT = float(FRCNN_CFG.get("TRAIN", "RPN_POSITIVE_WEIGHT"))

TRAIN_RPN_NMS_THRESH = float(FRCNN_CFG.get("TRAIN", "RPN_NMS_THRESH"))
TRAIN_RPN_PRE_NMS_TOP_N = int(FRCNN_CFG.get("TRAIN", "RPN_PRE_NMS_TOP_N"))
TRAIN_RPN_POST_NMS_TOP_N = int(FRCNN_CFG.get("TRAIN", "RPN_POST_NMS_TOP_N"))
TRAIN_RPN_MIN_SIZE = int(FRCNN_CFG.get("TRAIN", "RPN_MIN_SIZE"))

LEARNING_RATE = float(FRCNN_CFG.get("TRAIN", "LEARNING_RATE"))
WEIGHT_DECAY = float(FRCNN_CFG.get("TRAIN", "WEIGHT_DECAY"))
MOMENTUM = float(FRCNN_CFG.get("TRAIN", "MOMENTUM"))
ROI_BATCH_SIZE = int(FRCNN_CFG.get("TRAIN", "ROI_BATCH_SIZE"))
IMG_BATCH_SIZE = int(FRCNN_CFG.get("TRAIN", "IMG_BATCH_SIZE"))


TEST_RPN_NMS_THRESH = float(FRCNN_CFG.get("TEST", "RPN_NMS_THRESH"))
TEST_RPN_PRE_NMS_TOP_N = int(FRCNN_CFG.get("TEST", "RPN_PRE_NMS_TOP_N"))
TEST_RPN_POST_NMS_TOP_N = int(FRCNN_CFG.get("TEST", "RPN_POST_NMS_TOP_N"))
TEST_RPN_MIN_SIZE = int(FRCNN_CFG.get("TEST", "RPN_MIN_SIZE"))

ANCHOR_SCALES = FRCNN_CFG.get("LAYER", "ANCHOR_SCALES")
ANCHOR_FEAT_STRIDE = int(FRCNN_CFG.get("LAYER", "ANCHOR_FEAT_STRIDE"))
