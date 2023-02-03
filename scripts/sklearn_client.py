import time

import flwr as fl
from sklearn.model_selection import train_test_split


class SklearnClient(fl.client.NumPyClient):
    """
    This class if for creating Scikit-Learn clients for Flower.

    Args:
        model: The Scikit-Learn model. Must have the attribute coef_!
        X: All exogene variables used in this client.
        Y: All endogene variables used in this client.
        loss: The loss function this client uses.
        test_size: Percentage of data used for testing.
    """

    def __init__(self, model, X, Y, loss, test_size: float) -> None:
        super().__init__()
        self.model = model
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size)
        self.loss = loss


    def get_parameters(self, config):
        return self._get_model_params(self.model)

    def fit(self, parameters, config):
        #Set the model parameters
        self._set_model_params(self.model, parameters)

        #Train model
        self.model.fit(self.X_train, self.Y_train)

        return self._get_model_params(self.model), len(self.X_train), {} #Last argument is for the used metrics TODO implement logic

    def evaluate(self, parameters, config):
        self._set_model_params(self.model, parameters)
        loss = self.loss(self.Y_test, self.model.predict(self.X_test))

        return loss, len(self.X_test), {"accuracy": self.model.score(self.X_test, self.Y_test), "time": time.perf_counter()}

    def info(self):
        print(f'Class: {self.model} \nNumber of training samples: {len(self.X_train)}\nNumber of testing samples: {len(self.X_test)}')


    def _get_model_params(self, model):
        try:
            params = [
                    model.coef_,
                    model.intercept_
                ]
        except AttributeError:
            params = [
                model.coef_
            ]
        return params


    def _set_model_params(self, model, params):
        if hasattr(model, "intercept_"):
            model.intercept_ = params[1]
        model.coef_ = params[0]

        return model
