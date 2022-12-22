import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split


class TFClient(fl.client.NumPyClient):
    """
    This class if for creating Tensorflow clients for Flower.

    Args:
        model (tf.keras.Model): The Tensorflow model. Must already be compiled!
        X_train: Exogene variables used for training.
        Y_train: Endogene variables used for training.
        X_test: Exogene variables used for testing.
        Y_test: Endogene variables used for testing.
    """

    def __init__(self, model: tf.keras.Model, X_train, Y_train, epochs: int, test_size: float) -> None:
        super().__init__()
        self.model = model
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train, Y_train, test_size=test_size)
        self.epochs = epochs

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=32, steps_per_epoch=3)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": float(accuracy)}

    def info(self):
        self.model.summary()
        print(f'Number of training samples: {len(self.X_train)}\nNumber of testing samples: {len(self.X_test)}\nEpochs: {str(self.epochs)}')