import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import hyperparameters as hp


class Data():

    def __init__(self):
        self.b1 = os.listdir("../data/Train/BENIGN")
        self.b1 = self.set_filepaths(self.b1, "../data/Train/BENIGN/")
        self.b2 = os.listdir("../data/Test/BENIGN")
        self.b2 = self.set_filepaths(self.b2, "../data/Test/BENIGN/")

        self.m1 = os.listdir("../data/Train/MALIGNANT")
        self.m1 = self.set_filepaths(self.m1, "../data/Train/MALIGNANT/")
        self.m2 = os.listdir("../data/Test/MALIGNANT")
        self.m2 = self.set_filepaths(self.m2, "../data/Test/MALIGNANT/")

        self.n1 = os.listdir("../data/Train/NORMAL")
        self.n1 = self.set_filepaths(self.n1, "../data/Train/NORMAL/")
        self.n2 = os.listdir("../data/Test/NORMAL")
        self.n2 = self.set_filepaths(self.n2, "../data/Test/NORMAL/")



        self.images = np.concatenate([self.b1, self.b2, self.m1, self.m2, self.n1, self.n2])
        self.labels = np.concatenate(
            [np.full(len(self.b1) + len(self.b2),2),
                                            np.full(len(self.m1) + len(self.m2),1), np.full(len(self.n1) + len(self.n2),0)])
        
        self.data_sample = np.zeros((len(self.images),hp.img_size,hp.img_size,3))

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
    

    def set_filepaths(self, image_names, parent_path):
        for i in range(len(image_names)):
            image_names[i] = parent_path + image_names[i]
        
        return image_names


   
    def normalize(self):
        for j,filepath in enumerate(self.images): # j is index, i is file path
        
            img = Image.open(filepath)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255. # normalizing pixels to 0-1

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            self.data_sample[j] = img


    def preproccess(self):
        #using vgg16 preprocessing
        for i in range(len(self.data_sample)):
            self.data_sample[i] = tf.keras.applications.vgg16.preprocess_input(self.data_sample[i])


    def split_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.data_sample, 
                                                self.labels, test_size=0.25, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

