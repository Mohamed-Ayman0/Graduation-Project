import cv2
from cv2 import VideoCapture
import numpy as np
import  tensorflow as tf
from PIL import Image


from tensorflow.keras.preprocessing.image import load_img, img_to_array 

classes = ['glass', 'metal', 'paper', 'plastic']
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
cap = VideoCapture(1)
savedModel=tf.keras.models.load_model('model.h5')

while True:
    ret,frame = cap.read()

    img  = cv2.resize(frame,(256,256))
    #img  = load_img("images/paper60_r.jpg")
    #img  = img_to_array(img)
    #print(img.shape)
    img = np.reshape(img,[1,256,256,3])
    img = img/255
    prediction = savedModel.predict(img)
    print(f'the predection is :{str(classes[np.argmax(prediction)])}')
    cv2.imshow('detection',frame)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
cv2.destroyAllWindows()