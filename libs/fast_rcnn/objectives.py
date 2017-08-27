# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# --------------------------------------------------------

import tensorflow as tf
from libs.fast_rcnn.config import TRAIN_DEBUG
from keras import backend as Kb
import numpy as np

def softmaxWithLoss(name):
    print 'objectives: {}'.format(name)
    def softmaxWithLoss_sub(y_true, y_pred):
        #y_pred = np.array([Kb.get_value(y_pred)])
        print '{}: {}'.format(name, y_pred)
        return y_pred

    return softmaxWithLoss_sub

def smoothL1Loss(name):
    print 'objectives: {}'.format(name)
    def smoothL1Loss_sub(y_true, y_pred):
        #y_pred = np.array([Kb.get_value(y_pred)])
        print '{}: {}'.format(name, y_pred)
        return y_pred

    return smoothL1Loss_sub



