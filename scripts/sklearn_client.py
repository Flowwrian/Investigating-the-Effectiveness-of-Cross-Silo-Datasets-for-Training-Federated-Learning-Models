import flwr as fl

class SklearnClient(fl.client.NumPyClient):
    """
    This class if for creating Scikit-Learn clients for Flower.

    Args:
        model: The Scikit-Learn model. Must have the attribute coef_!
        X_train: Exogene variables used for training.
        Y_train: Endogene variables used for training.
        X_test: Exogene variables used for testing.
        Y_test: Endogene variables used for testing.
    """

    #__init__ needed for modularity
    def __init__(self, model, X_train, Y_train, X_test, Y_test, loss) -> None:
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.loss = loss

    def get_parameters(self, config):
        return _get_model_params(self.model)

    def fit(self, parameters, config):
        #Set the model parameters
        _set_model_params(self.model) #Use self.model =  _set_model_params(self.model) instead?

        #Train model
        self.model.fit(self.X_train, self.Y_train)

        return _get_model_params(self.model), len(self.X_train), {} #Last argument is for the used metrics TODO implement logic

    def evaluate(self, parameters, config):
        _set_model_params(self.model, parameters)
        return self.loss(self.Y_test, self.model.predict_proba(self.X_test)), len(self.X_test), {"accuracy": self.model.score(self.X_test, self.Y_test)}

    def info(self):
        print(f'Class: {self.model} \nNumber of training samples: {len(self.X_train)}\nNumber of testing samples: {len(self.X_test)}')


def _get_model_params(model):
    if hasattr(model, "intercept_"):
            params = [
                model.coef_,
                model.intercept_
            ]
    else:
        params = [
            model.coef_
        ]
    return params


def _set_model_params(model, params):
    if hasattr(model, "intercept_"):
        model.intercept_ = params[1]
    model.coef_ = params[0]

    return model