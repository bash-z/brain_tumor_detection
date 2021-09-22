from model import Model
import hyperparameters as hp
from preprocess import Data
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np


from tensorflow.keras.callbacks import ReduceLROnPlateau



def train(model, path_to_weights):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2)
    # callback_list=[reduce_lr,
    #         tf.keras.callbacks.TensorBoard(
    #         log_dir='logs',
    #         update_freq='epoch',
    #         profile_batch=0)]
    callback_list = [reduce_lr, tf.keras.callbacks.ModelCheckpoint(
        filepath=path_to_weights,
        save_weights_only=True
    )]


    history = model.fit(
        x=data.X_train,
        y=data.y_train,
        epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        shuffle=True,
        validation_split=0.20,
        verbose=1,
        callbacks=callback_list
    )

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # plt.savefig('loss.png')

def test(model):
    model.evaluate(
        x=data.X_test,
        y=data.y_test,
        verbose=1
    )

def interpret(image, label, model):
    # tf.compat.v1.enable_eager_execution()
    input = tf.Variable(Data._preprocess(Data._normalize(image)))
    with tf.GradientTape() as tape:
        # tape.watch(input)
        prediction = model(tf.expand_dims(input, axis=0), training=False)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(label, prediction)
        # loss = tf.losses.MSE(prediction, 1.0)
        # tf.losses.MSA()

    print("LOSS")
    print(loss)
    print(type(loss))
    print("INPUT")
    print(input)
    # print(type(input))
    gradients = tape.gradient(loss, input)
    print(gradients)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    x = axes[0].imshow(input.numpy(), cmap="gray")
    y = axes[1].imshow(np.squeeze(gradients) * input.numpy(), cmap="gray")
    plt.savefig('interpretation.png')





if __name__ == "__main__":
    data = Data()
    data.normalize()
    data.preprocess()
    data.split_data()

    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    

    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"]
        )

    print("TRAINING")
    train(model=model, path_to_weights=os.path.join(os.getcwd(), 'model_weights'))
    print("TESTING")
    test(model)
    print("INTERPRETATION")
    interpret(image="../data/Train/MALIGNANT/0.jpg", label=1, model=model)

