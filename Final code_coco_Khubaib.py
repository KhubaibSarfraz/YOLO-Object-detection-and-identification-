import cv2 as cv
import numpy as np


#define the confidence threshold
confThreshold= 0.6
#define non-max supression threshold
nmsThreshold= 0.40

inpWidth=416
inpHeight=416


#define classes file and names
classesFile= "coco_obj.names.txt"
classes=None

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#model config
modelConf= 'yolov3_coco.cfg.txt'

#define weights
modelWeights= 'yolov3-obj_7300.weights'

#setup NN
net= cv.dnn.readNetFromDarknet(modelConf,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputNames(net):
    layerNames= net.getLayerNames()

    return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(frame, outs):
    frameHeight= frame.shape[0]
    frameWidth=frame.shape[1]
 #bounding boxes according to confidence level
    classIDs = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
               
#perform non-max supprrsion on bounding boxes
                
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
#drawing bounding boxes
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)


def drawPred(classID, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0,255,0))

    label = '%.2f' % conf
    
    if classes:
        assert (classID < len(classes))
        label = '%s:%s' % (classes[classID], label)
    
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1]) 
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    
#setup for using webcam
winName= 'Real-time detection with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName,1000,1000)

cap = cv.VideoCapture(0)
while cv.waitKey(1) < 0:

    #get frame from video
    hasFrame, frame = cap.read()
    #Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs=net.forward(getOutputNames(net))
    postprocess(frame,outs)
    cv.imshow(winName,frame)

#

