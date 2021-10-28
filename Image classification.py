#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


# In[2]:


DIRECTORY= r"C:\Users\Asus\leaf"
CATAGORIES= ['Strawberry_fresh','Strawberry_scrotch']


# In[3]:



data=[]

for categories in CATAGORIES:
    folder=os.path.join(DIRECTORY,categories)
    label=CATAGORIES.index(categories)
    
    
    for img in os.listdir(folder):
        img=os.path.join(folder,img)
        img_arr=cv2.imread(img)
        img_arr=cv2.resize(img_arr,(100,100))
        
        data.append([img_arr,label])
        
        
        


# In[4]:


data


# In[5]:


random.shuffle(data)


# In[6]:


x=[]
y=[]


for features,label in data:
    x.append(features)
    y.append(label)


# In[7]:


X= np.array(x)
Y=np.array(y)


# In[8]:


x


# In[9]:


X=X/255


# In[10]:


X


# In[11]:


X.shape


# In[12]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation


# In[23]:


model=Sequential()
model.add( Conv2D(64,(3,3),input_shape=X.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add( Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(2,activation='softmax'))


# In[24]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[25]:


model.fit(X,Y,epochs=15,validation_split=0.1)


# In[26]:


model.summary()


# In[ ]:





# In[ ]:





# In[17]:


# Model Prediction


# In[27]:


from keras.preprocessing import image
import numpy as np

img_pred=image.load_img(r"C:\Users\Asus\leaf\Strawberry_scrotch\1a0eaf21-1ae4-4018-807a-af8667ed0811___RS_L.Scorch 0064.JPG",target_size=(100,100))

img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred, axis=0)


rslt= model.predict(img_pred)

print(rslt)
if rslt[0][0]>rslt[0][1]:
    prediction="Strawberry_fresh"
    
    
else:
    prediction="Strawberry_scrotch"
print(prediction)


# In[31]:


columns=8
rows=8

for ftr in rslt:
    fig=plt.figure(figsize=(12,12))
    for i in range(1,columns*rows+1):
        fig=plt.subplot(rows,columns,i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(ftr[0])
        plt.show


# In[ ]:




