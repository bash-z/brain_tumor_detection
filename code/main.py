import hyperparameters as hp
import preprocess



def train(model, datasets):
    """ Training routine. """

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
    )