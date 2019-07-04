import os 
import cv2
import pickle
from PIL import Image
import numpy as np

file_dir = os.path.dirname(__file__)
imgs_dir = os.path.join(file_dir,"images")

label_id_dict = dict()
label_count = 0

face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

train_datas = []
train_label_ids = []

for root, folders, filenames in os.walk(imgs_dir):
    if len(filenames)!=0:
        for f in filenames:
            if f.endswith('png') or f.endswith('jpg'):
                
                img_path = os.path.join(root, f)
                label = os.path.basename(root)
                if label not in label_id_dict.keys():
                    label_id_dict[label] = label_count
                    label_count += 1
                
                label_id = label_id_dict[label]

                formal_size = (168, 168)
                train_img = Image.open(img_path).convert("L")   # open this image and convert to gray scale
                resized_img = train_img.resize(formal_size, Image.ANTIALIAS)
                img_array = np.array(resized_img, 'uint8')        # tranform to numpy array

                faces = face_cascade.detectMultiScale(img_array, 1.3, 5)
                for face in faces:
                    (xi, yi, w, h) = face
                    face_region = img_array[yi:yi+h, xi:xi+w]   #get the detected face region
                    
                    train_datas.append(face_region)     # collect training data
                    train_label_ids.append(label_id)

with open('label_ids.pickle', 'wb') as f:
    pickle.dump(label_id_dict, f)

recognizer.train(train_datas, np.array(train_label_ids))
recognizer.save('recognize.yml')