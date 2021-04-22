import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Activation

import hyperparameters as hp

class Model(tf.keras.Model):
    """ Subclassing the model """

    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        # self.optimizer = tf.keras.optimizers.RMSprop(
        #     learning_rate=hp.learning_rate,
        #     momentum=hp.momentum)

        # self.architecture = [
        #     # Block 1
        #     Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        #     Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        #     MaxPool2D(pool_size=2),

        #     #Block 2
        #     Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
        #     Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
        #     MaxPool2D(pool_size=2),

        #     #Block 3
        #     Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
        #     MaxPool2D(pool_size=2),

        #     # Block 4
        #     Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
        #     MaxPool2D(pool_size=2),

        #     Dropout(0.2),
        #     Flatten(),
        #     Dense(units=128, activation="relu"),
        #     Dense(units=64, activation="relu"),
        #     Dropout(0.1),
        #     Dense(units=32, activation="relu"),
        #     Dense(units=hp.num_classes, activation="softmax")
        # ]

        ###MAIN
        self.architecture = [
            # Block 1
            Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),
            BatchNormalization(),

            #Block 2
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),
            BatchNormalization(),

            #Block 3
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),
            BatchNormalization(),

            Dropout(0.2),

            #Block 4
            Conv2D(filters=1028, kernel_size=3, padding="same", activation="relu"),
            Conv2D(filters=1028, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2),
            BatchNormalization(),

            Dropout(0.2),
            Flatten(),
            Dense(units=128, activation="relu"),
            Dense(units=64, activation="relu"),
            # Dropout(0.1),

            Dense(units=32, activation="relu"),
            Dense(units=hp.num_classes, activation="softmax")
            # Dense(units=32, activation="relu"),
            # Dense(units=hp.num_classes, activation="sigmoid")
        ]


        # self.architecture = [
        #     # Input(),
        #     # ZeroPadding2D((2, 2)),
        #
        #     Conv2D(32, (7, 7), strides = (1, 1)),
        #     BatchNormalization(axis = 3, name = 'bn0'),
        #     Activation('relu'),
        #
        #     Conv2D(64, (7, 7), strides = (1, 1)),
        #     BatchNormalization(axis = 3, name = 'bn1'),
        #     Activation('relu'),
        #
        #     MaxPool2D((4, 4)),
        #     MaxPool2D((4, 4)),
        #     Flatten(),
        #     Dense(3, activation='sigmoid'),
        #
        # ]

        # self.architecture = [Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'),
        #     Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'),
        #     MaxPool2D(pool_size=(6,6), padding='same'),
        #     Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'),
        #     Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'),
        #     MaxPool2D(pool_size=(2,2), padding='same'),
        #     Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'),
        #     Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'),
        #     MaxPool2D(pool_size=(2,2), padding='same'),
        #     Dropout(0.2),
        #     Flatten(),
        #     Dense(256, activation='relu'),
        #     Dense(hp.num_classes, activation='softmax')]



    def call(self, x):
        """ Forward pass. """

        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function. """

        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(labels, predictions)
