import os
import numpy as np
import tensorflow as tf


y = os.listdir("../data/yes")
n = os.listdir("../data/no")

images = np.concatenate([y,n])


labels = np.concatenate([np.full(len(y),1),np.full(len(n),0)])



#using vgg16 preprocessing
for i in range(len(images)):
    images[i] = tf.keras.applications.vgg16.preprocess_input(images[i])
