import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import deeplab_resnet_sketchParse_r5 as sketch_parse

def create_maskrcnn_resnet50_fpn():
    num_classes = 2 #Background and object
    
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='COCO_V1')

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor with a new one for our number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def create_resnet_sketch_parse_r5():
    model = sketch_parse.Res_Deeplab()
    return model