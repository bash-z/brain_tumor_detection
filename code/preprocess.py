import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import hyperparameters as hp


class Data():

    def __init__(self):
        self.y = os.listdir("../data/yes") # array of names of images (strings)
        self.n = os.listdir("../data/no")

        
        self.images = np.concatenate([self.y,self.n])
        self.labels = np.concatenate([np.full(len(self.y),1),np.full(len(self.n),0)])
        self.data_sample = np.zeros((len(self.images),hp.img_size,hp.img_size,3))

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

   
    def normalize(self):
        for j,i in enumerate(self.images): # j is index, i is file path
            if j < len(self.y):
                filepath = "../data/yes/"
            else:
                filepath = "../data/no/"

            img = Image.open(filepath + i)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255. # normalizing pixels to 0-1

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            self.data_sample[j] = img


    def preproccess(self):
        #using vgg16 preprocessing
        for image in self.data_sample:
            image = tf.keras.applications.vgg16.preprocess_input(image)


    def split_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.data_sample, 
                                                self.labels, test_size=0.25, random_state=39)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

