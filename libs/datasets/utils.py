# --------------------------------------------------------
# kr_faster_rcnn
# 
# Copyright (c) 2017
# Written by Romyny
# Licensed under The MIT License
# --------------------------------------------------------

import os, glob
from scipy.sparse import csr_matrix
import xml.etree.ElementTree as ET
import numpy as np
from libs.fast_rcnn.config import *
import os.path as osp
import PIL.Image as pil_im
from keras.applications.imagenet_utils import preprocess_input

def load_im_data(path, roi_entry, 
                 target_size=None, dim_ordering='default'):
    if pil_im is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    im = pil_im.open(path)    
    if roi_entry['flipped']:
        im = im.transpose(pil_im.FLIP_LEFT_RIGHT)
        
    im = im.convert('RGB')
        
    # substract pixel means
    #im = preprocess_input(im, dim_ordering)
    
    gt_inds = np.where(roi_entry['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        
    #resize im to match with network input
    if target_size == None:
        target_size = (224,224)
        
    w, h = im.size
    s_w = float(target_size[0]) / float(w)
    s_h = float(target_size[1]) / float(h)
    gt_boxes[:, 0] = roi_entry['boxes'][gt_inds, 0]*s_w
    gt_boxes[:, 2] = roi_entry['boxes'][gt_inds, 2]*s_w
    gt_boxes[:, 1] = roi_entry['boxes'][gt_inds, 1]*s_h
    gt_boxes[:, 3] = roi_entry['boxes'][gt_inds, 3]*s_h
    gt_boxes[:, 4] = roi_entry['gt_classes'][gt_inds]
    im = im.resize(size=target_size, resample=pil_im.BILINEAR)
        
    return im, gt_boxes      
    

def load_pascalvoc_data(dataset_DIR, class_to_ind):
    ann_DIR = os.path.join(dataset_DIR, "Annotations")
    imSets_DIR = os.path.join(dataset_DIR, "ImageSets")
    trainval_file = os.path.join(imSets_DIR, "Main", TRAIN_FN)
    test_file = os.path.join(imSets_DIR, "Main", TEST_FN)
        
    train_an = {}
    train_fn = []
    with open(trainval_file) as in_f:
        for im_fn in in_f:
            im_fn = im_fn.split('\n')[0].split('\r')[0]
            anno_pn = os.path.join(ann_DIR,im_fn+".xml")
            p_anno = readXmlAnno(anno_pn, class_to_ind)
            train_an[im_fn] = p_anno
            train_fn.append(im_fn)
        in_f.close()
            
    test_an = {}
    test_fn = []
    with open(test_file) as in_f:
        for im_fn in in_f:
            im_fn = im_fn.split('\n')[0].split('\r')[0]
            anno_pn = os.path.join(ann_DIR,im_fn+".xml")
            p_anno = readXmlAnno(anno_pn, class_to_ind)
            test_an[im_fn] = p_anno
            test_fn.append(im_fn)
        in_f.close()
            
    return {"im_fn": {"train_fn": train_fn, "test_fn": test_fn},
            "im_an": {"train_an": train_an, "test_an": test_an}}
        

def readXmlAnno(anno_pn, class_to_ind):
    tree = ET.parse(anno_pn)
    root = tree.getroot()
        
    objs = root.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, len(class_to_ind)), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)
        
    for id_obj, obj in enumerate(objs):
        xmin = int(float(obj.find('bndbox').find('xmin').text)) - 1
        ymin = int(float(obj.find('bndbox').find('ymin').text)) - 1
        xmax = int(float(obj.find('bndbox').find('xmax').text)) - 1
        ymax = int(float(obj.find('bndbox').find('ymax').text)) - 1
        id_cls = class_to_ind[obj.find('name').text.strip()]
        
        boxes[id_obj,:] = [xmin, ymin, xmax, ymax]
        gt_classes[id_obj] = id_cls
        overlaps[id_obj, id_cls] = 1.0
        seg_areas[id_obj] = (xmax - xmin + 1) * (ymax - ymin + 1)
    
    #overlaps = csr_matrix(overlaps)
    
    return {"boxes": boxes,
            "gt_classes": gt_classes,
            "gt_overlaps": overlaps,
            "seg_areas": seg_areas,
            "flipped": False}

def writeXmlAnno(p_anno, anno_pn, folder):
    
    img_annotation = ""
    img_annotation += "<annotation>\n"
    img_annotation += "\t<folder>"+folder+"</folder>\n"
    img_annotation += "\t<filename>im_fn</filename>\n"
    img_annotation += "\t<source>\n"
    img_annotation += "\t\t<database>Unspecified</database>\n"
    img_annotation += "\t\t<annotation>Unspecified</annotation>\n"
    img_annotation += "\t\t<image>Unspecified</image>\n"
    img_annotation += "\t</source>\n"
    img_annotation += "\t<owner>\n"
    img_annotation += "\t\t<name>Unspecified</name>\n"
    img_annotation += "\t</owner>\n"
    img_annotation += "\t<size>\n"
    img_annotation += "\t\t<width>"+p_anno['size']['width']+"</width>\n"
    img_annotation += "\t\t<height>"+p_anno['size']['height']+"</height>\n"
    img_annotation += "\t\t<depth>"+p_anno['size']['depth']+"</depth>\n"
    img_annotation += "\t</size>\n"
    img_annotation += "\t<segmented>0</segmented>\n" 
    
    for obj in p_anno['l_obj']:
        img_annotation += "\t<object>\n"
        img_annotation += "\t\t<name>"+obj["name"]+"</name>\n"
        img_annotation += "\t\t<type>"+obj["type"]+"</type>\n"
        img_annotation += "\t\t<pose>Unspecified</pose>\n"
        img_annotation += "\t\t<truncated>1</truncated>\n"
        img_annotation += "\t\t<difficult>0</difficult>\n"
        img_annotation += "\t\t<bndbox>\n"
        img_annotation += "\t\t\t<xmin>"+str(obj["xmin"])+"</xmin>\n"
        img_annotation += "\t\t\t<ymin>"+str(obj["ymin"])+"</ymin>\n"
        img_annotation += "\t\t\t<xmax>"+str(obj["xmax"])+"</xmax>\n"
        img_annotation += "\t\t\t<ymax>"+str(obj["ymax"])+"</ymax>\n"
        img_annotation += "\t\t</bndbox>\n"
        img_annotation += "\t</object>\n"
        
    img_annotation += "</annotation>"    
    with open(anno_pn, "w") as annotations_file:
        annotations_file.write(img_annotation)    
