from model import Model
import hyperparameters as hp
from preprocess import Data
import tensorflow as tf



def train(model):
    model.fit(
        x=data.X_train,
        y=data.y_train,
        epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        shuffle=True,
        validation_split=0.20,
        verbose=1
    )

def test(model):
    model.evaluate(
        x=data.X_test,
        y=data.y_test,
        verbose=1
    )


if __name__ == "__main__":
    data = Data()
    data.normalize()
    data.preproccess()
    data.split_data()

    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

    model.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"]
        )
    
    train(model)
    # test(model)
    

    

