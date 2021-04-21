from model import Model
import hyperparameters as hp
from preprocess import Data
import tensorflow as tf



def train(model, datasets):
    """ Training routine. """

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
    )


if __name__ == "__main__":
    data = Data()
    data.normalize()
    data.preproccess()
    X_train, X_test, y_train, y_test = data.split_data()

    model = Model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    train()
    

    

