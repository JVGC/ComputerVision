import numpy as np
import cv2
import time

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'frozen_inference_graph.pbtxt')
 
# Input image
img = cv2.imread('../../object-detection/object-detection-deep-learning/images/example_01.jpg')
rows = img.shape[0]
cols = img.shape[1]
 
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
 
labels = open('labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in labels]


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
        label = "{} {:.2f}%" .format(classes[index],score * 100)
        #draw a red rectangle around detected objects
        cv2.putText(img, label, (int(xStart), int(yStart)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(xStart), int(yStart)+15), (int(xEnd), int(yEnd)), (0, 0, 255), thickness=2)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()