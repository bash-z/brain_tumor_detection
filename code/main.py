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
    if not os.path.exists(os.path.join(os.getcwd(), 'figures')):
        os.makedirs(os.path.join(os.path.dirname(os.getcwd()), 'figures'))
    plot(history, 'loss', 'val_loss', os.path.join('../figures', 'loss.png'))
    plot(history, 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', os.path.join('../figures', 'accuracy.png'))
    

def plot(history, training_metric, validation_metric, filename):
    plt.plot(history.history[training_metric])
    plt.plot(history.history[validation_metric])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(filename)


def test(model):
    model.evaluate(
        x=data.X_test,
        y=data.y_test,
        verbose=1
    )


def interpret(image, label, model, filename):
    if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'interpretation')):
        os.makedirs(os.path.join(os.path.dirname(os.getcwd()), 'interpretation'))

    input = tf.Variable(Data._normalize(image))
    with tf.GradientTape() as tape:
        prediction = model(tf.expand_dims(input, axis=0), training=False)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(label, prediction)  
    gradients = tape.gradient(loss, input)

    print(f'INPUT: {input}')
    print(f'LOSS: {loss}')
    print(f'GRADIENTS: {gradients}')

    plt.style.use('grayscale')
    fig, axes = plt.subplots(nrows=1, ncols=2)
    x = axes[0].imshow(input.numpy())
    y = axes[1].imshow(np.squeeze(gradients) * input.numpy())
    plt.savefig(os.path.join('../interpretation', filename))


if __name__ == "__main__":
    data = Data()
    data.normalize()
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
    train(model=model, path_to_weights=os.path.join(os.path.dirname(os.getcwd()), 'model_weights'))
    print("TESTING")
    test(model)
    print("INTERPRETATION")
    for root, dirs, files in os.walk('../data/Train'):
        for f in files:
            type = os.path.basename(os.path.dirname(os.path.join(root, f)))
            if type == "BENIGN":
                label = 2
            elif type == "MALIGNANT":
                label = 1
            else:
                label = 0
            interpret(image=os.path.join(root, f), label=label, model=model, filename=type + "_" + f)