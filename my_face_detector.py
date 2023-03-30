"""
    A simple class for face detection using ResNet SSD detector model
"""


import os
import numpy as np
import cv2


class FaceDetector:
    def __init__(self, model_directory):
        '''
            model_directory: Path to folder containing "deploy.prototxt" 
            and "res10_300x300_ssd_iter_140000.caffemodel" files
        '''
        prototxtPath = os.path.join(model_directory, "deploy.prototxt")
        weightsPath = os.path.join(model_directory, "res10_300x300_ssd_iter_140000.caffemodel")
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    def detect(self, img, min_confidence=0.5):
        '''
            - img: RGB image with shape = (height, width, channels=3)
            - min_confidence: Minimum confidence (between 0 and 1) to detect a face
            - return: a list of dictionaries where each dictionary contains 3 keys:
                1. box: bounding box of a detected face
                2. img: sub-image of a detected face 
                3. confidence: Confidence of detection (between 0 and 1)
        '''
        # image pre-processing
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img, 
            scalefactor=1.0, 
            size=(224, 224),
            mean=(104.0, 177.0, 123.0),
        )
        # detect faces
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward().reshape(-1, 7)
        # remove weak detections using min_confidence
        detections = detections[ detections[:, 2] >= min_confidence ]
        # initialize result
        result = []
        # loop through detections
        for det in detections:
            # get confidence
            confidence = det[2]
            # get box (startX, startY, endX, endY)
            box = det[3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # validate box
            if (startX < 0 or startX >= w 
                    or endX < 0   or endX >= w 
                    or startY < 0 or startY >= h 
                    or endY < 0   or endY >= h
                    or endX <= startX
                    or endY <= startY):
                continue
            # slice face sub-image
            face = img[startY:endY, startX:endX]
            # calculate face width and height
            faceWidth = endX - startX
            faceHeight = endY - startY 
            # calculate face box (startX, startY, faceWidth, faceHeight)
            faceBox = (startX, startY, faceWidth, faceHeight)
            # append face to result
            result.append({
                'confidence': confidence,
                'box': faceBox,
                'img': face
            })
        return result