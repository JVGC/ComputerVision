import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe deploy 'prototxt' file")
ap.add_argument("-m", "--model", required = True, help = "path to caffe pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.2, help = "minimum probability to filter weak dectections")

args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

Colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("INFO: Carregando o Modelo")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv2.VideoCapture('videos/TownCentreXVID.avi')
ret, image = cap.read()
#print(type(image))

while(True):
    
    ret, image = cap.read()
    (h, w) = image.shape[:2]

    #param: (image, fator de escalação, size do blob, mean value que sera subtraido dos canais de cores)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300,300),127.5) 

    #print("INFO: Computando as detecções")
    net.setInput(blob)
    detections = net.forward()


    #detection[1, 1, num de objetos detectados, 
    # (0 não retorna nada; 1 retorna o indice da classe que foi detectada, 2 retorna a porcentagem dela,
    # de 3 a 7 temos as coordenadas da caixa (startX, startY, endX, endY)]


    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i, 2]

        if(confidence > args["confidence"]):
            index = int(detections[0,0,i,1])
            box = detections[0,0,i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%" .format(CLASSES[index], confidence * 100)
            print("INFO:{}". format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY), Colors[index], 2) #desenhando o retângulo na imagem
            y = startY-15 if startY-15 > 15 else startY+15 #determinando o Y de onde vai ficar a label do objeto
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors[index], 1)
    cv2.imshow("Image", image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()