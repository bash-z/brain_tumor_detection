import tensorflow as tf

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate)

        self.architecture = [
            # Block 1
            Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),

            #Block 2
            Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),

            #Block 3
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),

            # Block 4
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),

            Dropout(0.2),
            Flatten(),
            Dense(units=128, activation="relu"),
            Dense(units=64, activation="relu"),
            Dropout(0.1),
            Dense(units=32, activation="relu"),
            Dense(units=15, activation="softmax")
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(labels, predictions)

