# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# based on py-faster-rcnn
# --------------------------------------------------------

from libs.fast_rcnn.config import *
import libs.fast_rcnn.roi_data_layer.roidb as rdl_roidb

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if USE_FLIPPED:
        print '[INFO] Appending horizontally-flipped training examples: start'
        imdb.append_flipped_images()
        print '[INFO] Appending horizontally-flipped training examples: done'

    print '[INFO] Preparing training data: start'
    rdl_roidb.prepare_roidb(imdb)
    print '[INFO] Preparing training data: done'

    return imdb.roidb
