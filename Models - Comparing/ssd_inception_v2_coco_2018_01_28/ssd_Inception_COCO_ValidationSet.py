# python3 ssd_Inception_COCO_ValidationSet.py --weights frozen_inference_graph.pb --prototxt frozen_inference_graph.pbtxt --images val2017/ --annotations annotations/instances_val2017.json

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
net = cv.dnn.readNetFromTensorflow((args.weights), (args.prototxt))
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV);

labels = open('labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in labels]


detections = []
images = os.listdir(args.images)
j = 0
print(len(images))
for imgName in images:
    inp = cv.imread((os.path.join(args.images, imgName)))
    rows = inp.shape[0]
    cols = inp.shape[1]
    if(j % 500 == 0):
      print(j)
    j +=1  


    net.setInput(cv.dnn.blobFromImage(inp, size = (300, 300),swapRB = True))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        if score > 0.01: 
          classId = int(out[0, 0, i, 1])
         

          x = out[0, 0, i, 3] * cols
          y = out[0, 0, i, 4] * rows
          w = out[0, 0, i, 5] * cols - x
          h = out[0, 0, i, 6] * rows - y
          detections.append({
            "image_id": int(imgName.rstrip('0')[:imgName.rfind('.')]),
            "category_id": classId,
            "bbox": [x,y,w,h],
            "score": score
          })
    
with open('cv_result.json', 'wt') as f:
    json.dump(detections, f)