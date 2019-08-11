#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author = Pranshu Aggarwal
# Recognise faces using some classification algorithm - like logistic, knn, svm etc.

# 1. Read a video stream using opencv.
# 2. Extract faces out of it.
# 3. Load the training data (numpy arrays of all the persons)
    # x-values are stored in the numpy arrays
    # y-values we need to assign for each person
# 4. Use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user.
# 6. Display the predictions on the screen - bounding box and name

import cv2
import os
import numpy as np

########################################################

def distance(v1, v2):

    # Eucledian

    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):

        # Get the vector and label

        ix = train[i, :-1]
        iy = train[i, -1]

        # Compute the distance from test point

        d = distance(test, ix)
        dist.append([d, iy])

    # Sort based on distance and get top k predictions

    dk = sorted(dist, key=lambda x: x[0])[:k]

    # Retrieve only the labels

    labels = np.array(dk)[:, -1]

    # Get frequencies of each label

    output = np.unique(labels, return_counts=True)

    # Find max frequency and corresponding label

    index = np.argmax(output[1])

    return output[0][index]


##########################################################

# Init Camera

cap = cv2.VideoCapture(0)

# Face Detection

face_cascade = cv2.CascadeClassifier('face-detector.xml')

count = 0
dataset_path = './facedata/'

face_data = []
labels = []

class_id = 0  # Labels for the given file
names = {}  # Mapping between id-name

# Data Preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        # Create a mapping between class_id and name

        names[class_id] = fx.split('.')[0]

        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create Labels for the class

        target = class_id * np.ones((data_item.shape[0], ))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_labels), axis=1)

while True:
    (ret, frame) = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        (x, y, w, h) = face

        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x+ w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        # Display on the screen the name and rectangle around it

        pred_name = names[int(out)]

        # putText(frame,text,coordinates,fontType,fontScale,fontColor,thickness,lineType)

        cv2.putText(frame,pred_name,(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(0xFF, 0, 0),2,cv2.LINE_AA,)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0xFF, 0xFF), 2)

    cv2.imshow('Faces', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


			