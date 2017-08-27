# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romuald FOTSO
# Licensed under The MIT License
# based on py-faster-rcnn
# --------------------------------------------------------
from libs.datasets.IMDB import IMDB
from libs.datasets.utils import *
import os, cPickle

class PascalVoc(IMDB):
    """Image database related to pascal_voc"""
    def __init__(self, name, image_set, dataset_DIR, classes_file):
        #IMDB.__init__(name)
        self.__image_set = image_set        
        self._classes = []
        super(PascalVoc, self).__init__(name)
        self._dataset_DIR = dataset_DIR
        
        with open(classes_file, 'r') as f:
            for line in f:
                cls = line.split('\n')[0]
                self._classes.append(cls)
        f.close()
        
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        
        # load train & test data info
        pascalvoc_data = load_pascalvoc_data(self._dataset_DIR, self._class_to_ind)
        self._image_index = []
        if image_set == "trainval":
            self._image_index = pascalvoc_data["im_fn"]["train_fn"]
        elif image_set == "test":
            self._image_index = pascalvoc_data["im_fn"]["test_fn"]
        else:
            raise ("[ERROR] unknown image_set {}".format(image_set))
        
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

    @property
    def class_to_ind(self):
        return self._class_to_ind

    @property
    def min_rois_len(self):
        return self._min_rois_len

    @property
    def max_rois_len(self):
        return self._max_rois_len

    def find_rois_lens(self):
        self._max_rois_len = 0
        self._min_rois_len = 99999
        for idx, roi_entry in enumerate(self.roidb):
            rois_len = len(roi_entry["boxes"])
            if rois_len > self._max_rois_len:
                self._max_rois_len = rois_len
            if rois_len < self._min_rois_len:
                self._min_rois_len = rois_len

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '[INFO] {} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set == 'trainval':
            # load gth annotations from .xml files
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            #roidb = IMDB.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print '[INFO] wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        """
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)
        #"""
        raise NotImplementedError
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '[INFO] {} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb        
        
        gt_roidb = [readXmlAnno(osp.join(self._dataset_DIR,"Annotations",im_name+".xml"), 
                                self._class_to_ind) for im_name in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print '[INFO] wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    
    def filter_roidb(self, roidb):
        """Remove roidb entries that have no usable RoIs."""
    
        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < BG_THRESH_HI) &
                               (overlaps >= BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid    
        
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        
        return filtered_roidb
