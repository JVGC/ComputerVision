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

from __future__ import print_function
import numpy as np
import time
import cv2
import json
import argparse
import os

parser = argparse.ArgumentParser(
    description='Evaluate MobileNet-SSD model using both TensorFlow and OpenCV. '
                'COCO evaluation framework is required: http://cocodataset.org')
parser.add_argument('--weights', required=True,
                    help='Path to frozen_inference_graph.pb of MobileNet-SSD model. ')
parser.add_argument('--prototxt', help='Path to ssd_mobilenet_v1_coco.pbtxt from opencv_extra.', required=True)
parser.add_argument('--images', help='Path to COCO validation images directory.', required=True)
parser.add_argument('--annotations', help='Path to COCO annotations file.', required=True)
args = parser.parse_args()

tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'frozen_inference_graph.pbtxt')

img = cv2.imread('../../object-detection/object-detection-deep-learning/images/example_01.jpg')

rows = img.shape[0]
cols = img.shape[1]
 
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
 
""" labels = open('labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in labels] """


# Runs a forward pass to compute the net output
start = time.time()
networkOutput = tensorflowNet.forward()
end = time.time()
print("Tempo de detecção: {:.5} seconds".format(end - start))


# Loop on the outputs
for detection in networkOutput[0,0]:
    
    score = float(detection[2])
    if score > 0.3:
        index = int(detection[1])
        xStart = detection[3] * cols
        yStart = detection[4] * rows
        xEnd = detection[5] * cols
        yEnd = detection[6] * rows
        #label = "{} {:.2f}%" .format(classes[index],score * 100)
        label = "{:.2f}%" .format(score * 100)
        #draw a red rectangle around detected objects
        cv2.putText(img, label, (int(xStart), int(yStart)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(xStart), int(yStart)+15), (int(xEnd), int(yEnd)), (0, 0, 255), thickness=2)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()