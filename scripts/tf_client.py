import flwr as fl
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class TFClient(fl.client.NumPyClient):
    """
    This class if for creating Tensorflow clients for Flower.

    Args:
        model (tf.keras.Model): The Tensorflow model. Must already be compiled!
        X: All exogene variables used in this client.
        Y: All endogene variables used in this client.
        epochs (int): Number of training epochs.
        batch_size (int): Number of entries per batch.
        test_size (float): Percentage of test size.
    """

    def __init__(self, model: tf.keras.Model, X, Y, epochs: int, batch_size:int, test_size: float) -> None:
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size)
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.Y_train = np.array(self.Y_train)
        self.Y_test = np.array(self.Y_test)
        self.epochs = epochs

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": float(accuracy)}

    def info(self):
        self.model.summary()
        print(f'Number of training samples: {len(self.X_train)}\nNumber of testing samples: {len(self.X_test)}\nEpochs: {str(self.epochs)}')
