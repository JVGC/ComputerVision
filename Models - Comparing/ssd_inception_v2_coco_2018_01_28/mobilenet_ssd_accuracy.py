from __future__ import print_function
# Script to evaluate MobileNet-SSD object detection model trained in TensorFlow
# using both TensorFlow and OpenCV. Example:
#
# python mobilenet_ssd_accuracy.py \
#   --weights=frozen_inference_graph.pb \
#   --prototxt=ssd_mobilenet_v1_coco.pbtxt \
#   --images=val2017 \
#   --annotations=annotations/instances_val2017.json
#
# Tested on COCO 2017 object detection dataset, http://cocodataset.org/#download
# python3 mobilenet_ssd_accuracy.py
import os
import cv2 as cv
import json
import argparse

### Evaluation part ############################################################

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = 'bbox'
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
cocoGt=COCO('annotations/instances_val2017.json')

#initialize COCO detections api
for resFile in ['cv_result.json']:
    print(resFile)
    cocoDt=cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()