from model import Model
import hyperparameters as hp
from preprocess import Data
import tensorflow as tf
import os
import datetime
from matplotlib import pyplot as plt
# from tensorboard_utils import \
#         ImageLabelingLogger, CustomModelSaver

from tensorflow.keras.callbacks import ReduceLROnPlateau



def train(model, checkpoint_path, logs_path):


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2)
    callback_list=[reduce_lr,
            tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            update_freq='epoch',
            profile_batch=0)]
    # callback_list = [
    #     tf.keras.callbacks.TensorBoard(
    #         log_dir='logs',
    #         update_freq='epoch',
    #         profile_batch=0)
    #     # ImageLabelingLogger(logs_path, data),
    #     # CustomModelSaver(checkpoint_path, 1, hp.max_num_weights)
    # ]


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
    plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('loss.png')

def test(model):
    model.evaluate(
        x=data.X_test,
        y=data.y_test,
        verbose=1
    )


if __name__ == "__main__":
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    data = Data()
    data.normalize()
    data.preproccess()
    data.split_data()

    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"]
        )

    train(model, checkpoint_path, logs_path)
    # test(model)
