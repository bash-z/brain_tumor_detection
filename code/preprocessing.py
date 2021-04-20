import os
import numpy as np
import tensorflow as tf
from PIL import Image


y = os.listdir("../data/yes")
n = os.listdir("../data/no")

images = np.concatenate([y,n])


labels = np.concatenate([np.full(len(y),1),np.full(len(n),0)])

data_sample = np.zeros((len(images),32,32,3))


for j,i in enumerate(y):
    img = Image.open("../data/yes/" + i)
    img = img.resize((32, 32))
    img = np.array(img, dtype=np.float32)
    img /= 255.

    # Grayscale -> RGB
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    #print(img.shape)

    data_sample[j] = img


for j,i in enumerate(n):
    img = Image.open("../data/no/" + i)
    img = img.resize((32, 32))
    img = np.array(img, dtype=np.float32)
    #print(img.shape)
    img /= 255.
    #print(img.shape)

    # Grayscale -> RGB
    if len(img.shape) == 2:
        #print('check')
        img = np.stack([img, img, img], axis=-1)

    #print(img.shape)
    #print(i)
    data_sample[j] = img




#using vgg16 preprocessing
for i in range(len(data_sample)):
    data_sample[i] = tf.keras.applications.vgg16.preprocess_input(data_sample[i])
    
print(data_sample.shape)
#print(np.asarray(data_sample).shape)
