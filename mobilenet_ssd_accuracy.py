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
# python3 mobilenet_ssd_accuracy.py --weights ./ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --prototxt ./ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pbtxt --images val2017/ --annotations annotations/instances_val2017.json
import os
import cv2 as cv
import json
import argparse

parser = argparse.ArgumentParser(
    description='Evaluate MobileNet-SSD model using both TensorFlow and OpenCV. '
                'COCO evaluation framework is required: http://cocodataset.org')
parser.add_argument('--weights', required=True,
                    help='Path to frozen_inference_graph.pb of MobileNet-SSD model. '
                         'Download it from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
parser.add_argument('--prototxt', help='Path to ssd_mobilenet_v1_coco.pbtxt from opencv_extra.', required=True)
parser.add_argument('--images', help='Path to COCO validation images directory.', required=True)
parser.add_argument('--annotations', help='Path to COCO annotations file.', required=True)
args = parser.parse_args()


### Get OpenCV predictions #####################################################
net = cv.dnn.readNetFromTensorflow(cv.samples.findFile(args.weights), cv.samples.findFile(args.prototxt))
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV);

detections = []
for imgName in os.listdir(args.images):
    inp = cv.imread(cv.samples.findFile(os.path.join(args.images, imgName)))
    rows = inp.shape[0]
    cols = inp.shape[1]
    inp = cv.resize(inp, (300, 300))

    net.setInput(cv.dnn.blobFromImage(inp, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), True))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        # Confidence threshold is in prototxt.
        classId = int(out[0, 0, i, 1])

        x = out[0, 0, i, 3] * cols
        y = out[0, 0, i, 4] * rows
        w = out[0, 0, i, 5] * cols - x
        h = out[0, 0, i, 6] * rows - y
        detections.append({
          "image_id": int(imgName.rstrip('0')[:imgName.rfind('.')]),
          "category_id": classId,
          "bbox": [x, y, w, h],
          "score": score
        })

with open('cv_result.json', 'wt') as f:
    json.dump(detections, f)

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
cocoGt=COCO(args.annotations)

#initialize COCO detections api
for resFile in ['tf_result.json', 'cv_result.json']:
    print(resFile)
    cocoDt=cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()