#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D
import pickle 


# In[2]:


images = []
classNo = []
path = "num"
for i in range(0, 10):
    mylist = os.listdir(path + "/"  + str(i))
    for j in mylist:
        img = cv.imread(path + "/" + str(i) + "/" + j)
        img = cv.resize(img, (32,32))
        images.append(img)
        classNo.append(i)


# In[3]:


myList = os.listdir(path)
noofclasses = len(myList)
noofclasses


# In[4]:


images = np.array(images)
classNo = np.array(classNo)


# In[5]:


images.shape


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


xtrain, xtest, ytrain, ytest = train_test_split(images, classNo,train_size=0.7, test_size=0.3, random_state=42)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.3)


# In[8]:


xtrain.shape


# In[9]:


def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img


# In[10]:


xtrain = np.array(list(map(preprocess, xtrain)))
xtest = np.array(list(map(preprocess, xtest)))
xvalid = np.array(list(map(preprocess, xvalid)))


# In[11]:


xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2], 1)
xvalid = xvalid.reshape(xvalid.shape[0], xvalid.shape[1], xvalid.shape[2], 1)


# In[12]:


xtrain.shape


# In[13]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[14]:


xtrain = xtrain.reshape(-1, 32, 32, 1)


# In[15]:


datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            shear_range=0.1,
            rotation_range=10
            )
datagen.fit(xtrain)


# In[16]:


from tensorflow.keras.utils import to_categorical


# In[17]:


ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest, 10)
yvalid = to_categorical(yvalid, 10)


# In[18]:


ytrain.shape


# In[19]:


xtrain.shape


# In[20]:


def myModel():
    nooffilters = 60
    sizeoff1 = (5,5)
    sizeoff2 = (3,3)
    sizeofPool = (2,2)
    noofNode = 500
    
    model = Sequential()
    model.add((Conv2D(nooffilters, sizeoff1, input_shape = (32,32,1), activation="relu")))
    model.add((Conv2D(nooffilters, sizeoff1, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    
    model.add((Conv2D(nooffilters//2, sizeoff2,  activation="relu")))
    model.add((Conv2D(nooffilters//2, sizeoff2, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(noofNode, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(noofclasses, activation="softmax"))
    model.compile(Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
    
    return model


# In[21]:


model = myModel()


# In[22]:


model.summary()


# In[23]:


yvalid.shape


# In[30]:


batchsize = 10
epochsval = 10
# steps_per_epoch = 2000
# num_batches = len(xtrain) // batchsize
# print(num_batches)
# Ensure steps_per_epoch does not exceed num_batches
# steps_per_epoch = len(xtrain)//batchsize
# train_generator = datagen.flow(xtrain, ytrain, batch_size=batchsize)
# for epoch in range(epochsval):
#     train_generator.reset()
#     history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=1, 
#                     validation_data=(xvalid, yvalid), shuffle=True)
history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batchsize), epochs = epochsval, validation_data=(xvalid, yvalid), shuffle=1, verbose=1)


# In[31]:


history.history['val_loss']


# In[32]:


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
score = model.evaluate(xtest, ytest)
print(score[0])
print(score[1])


# In[33]:


pickle_out = open("model_training.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[ ]:




