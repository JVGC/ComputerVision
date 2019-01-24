import numpy as np 
import time
import cv2

tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'frozen_inference_graph.pbtxt')

labels = open('labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in labels]
soma = cont = 0
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    #img = cv2.imread('../../object-detection/object-detection-deep-learning/images/example_01.jpg')

    rows, cols = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, size = (300,300), swapRB = True, crop = False)

    tensorflowNet.setInput(blob)

    start = time.time()
    detections = tensorflowNet.forward()
    end =  time.time()

    soma = soma + (end-start)
    cont = cont+1
    print("Tempo de detecção = {:.2}" .format(end-start))

    for detection in (detections[0,0]):

        score = detection[2]
        if(score > 0.5):
            index = int(detection[1])
            Xstart = detection[3] * cols
            Ystart = detection[4] * rows
            Xend = detection[5] * cols
            Yend = detection[6] * rows
            label = "{} {:.2f}%" .format(classes[index-1],score * 100)
            #label = "{:.2f}%" .format(score*100)
            cv2.putText(img, label, (int(Xstart), int(Ystart)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(Xstart), int(Ystart)), (int(Xend), int(Yend)), (0,0,255),thickness = 2)

    cv2.imshow("Image", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

print("cont = {}" .format(cont))
print("Media = {:.2} segundos" .format(soma/cont))
cap.release()
cv2.destroyAllWindows()