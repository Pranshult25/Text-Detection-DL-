#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle 
import cv2 as cv


# In[2]:


cam = cv.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)


# In[3]:


pickle_in = open("model_training.p", "rb")
model = pickle.load(pickle_in)


# In[4]:


def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img


# In[5]:


while True:
    _, img = cam.read()
    imgorg = img
    img = np.asarray(img)
    img = cv.resize(img, (32,32))
    img = preprocess(img)
    cv.imshow("Frame", img)
    img = img.reshape(1,32,32,1)
#     print(img.shape)
#     val = int(model.predict_classes(img))
    predicts = model.predict(img)
    val = np.argmax(predicts, axis=1)
    probval = np.amax(predicts)
    if(probval > 0.65):
        print(val, probval)
        cv.putText(imgorg, str(val) + "  " + str(probval), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow("Image", imgorg)
    if(cv.waitKey(1) == 27):
        break
cam.release()
cv.destroyAllWindows()


# In[ ]:




