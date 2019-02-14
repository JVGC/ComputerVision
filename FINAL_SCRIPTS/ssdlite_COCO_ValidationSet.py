# Rodar como:
# python3 ssdlite_COCO_ValidationSet.py --weights frozen_inference_graph.pb --prototxt frozen_inference_graph.pbtxt --images val2017/ --annotations annotations/instances_val2017.json

def read_classes():
    labels = open('labels.txt').read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in labels]
    return classes

import os
import cv2 as cv
import json
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True,
                    help='Caminho para frozen_inference_graph.pb o', required = True)
parser.add_argument('--prototxt', help='Caminho para ssd_mobilenet_v1_coco.pbtxt from opencv_extra.', required=True)
parser.add_argument('--images', help='Caminho para COCO validation images directory.', required=True)
parser.add_argument('--annotations', help='Caminho para COCO annotations file.', required=True)
args = parser.parse_args()


# Lendo a rede. NOTA: Se o arquivo ".pbtxt" não vier, utilizar o arquivo "tf_text_graph_ssd.py" para gerá-lo
net = cv.dnn.readNetFromTensorflow((args.weights), (args.prototxt))
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV);

classes = read_classes()


detections = []
# Criando uma lista com todas as imagens
images = os.listdir(args.images)

for imgName in images:
    inp = cv.imread((os.path.join(args.images, imgName)))
    rows, cols = inp.shape[:2]

    # O modelo SSD utiliza imagens de tamanho (300,300)
    net.setInput(cv.dnn.blobFromImage(inp, size = (300, 300),swapRB = True))

    start = time.time()
    out = net.forward()
    end = time.time()
    # Passando por todos os objetos detectados e limitando o score acima de 1%
    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        # Passando por cada detecção e guardando os valores necessários de acordo com o COCO
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