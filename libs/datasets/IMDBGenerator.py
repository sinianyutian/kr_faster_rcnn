# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# --------------------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
from libs.datasets.PascalVocIterator import PascalVocIterator


class IMDBGenerator(ImageDataGenerator):
    
    def __init__(self,imdb,roidb,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 dim_ordering='default'):
        self._imdb = imdb
        self._roidb = roidb
        
        super(IMDBGenerator, self).__init__(featurewise_center,
                 samplewise_center,
                 featurewise_std_normalization,
                 samplewise_std_normalization,
                 zca_whitening,rotation_range,
                 width_shift_range,height_shift_range,
                 shear_range,zoom_range,
                 channel_shift_range,fill_mode,cval,
                 horizontal_flip,vertical_flip,
                 rescale,preprocessing_function,
                 dim_ordering)

    @property
    def imdb(self):
        return self._imdb

    @property
    def roidb(self):
        return self._roidb

    def flow_from_pascalvoc(self, directory,
                            target_size=(224, 224), color_mode='rgb',
                            dim_ordering='default',
                            batch_size=1, seed=None):
        return PascalVocIterator(
                directory, self,
                target_size=(224, 224), color_mode='rgb',
                dim_ordering='default',
                batch_size=1, seed=None)
    
    
