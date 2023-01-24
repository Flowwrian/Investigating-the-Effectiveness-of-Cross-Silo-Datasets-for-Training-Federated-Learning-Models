from pathlib import Path
import pickle

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sktime.forecasting.model_selection import SlidingWindowSplitter

from sklearn_client import SklearnClient
from tf_client import TFClient


def get_samples(dataset: str, n: int, x_attributes: list[str], station: str, serialize: bool, max_samples: int = 100000):
    """
    Generate samples from a given dataset. The samples are created by using a rolling window.
    Args:
        data (str): The dataset sampling from. Can be 'covid' or 'weather'.
        n (int): Number of records per sample.
        attributes (list[str]): A list of all variables used from the dataset. The first entry is the target value (endogene variable), as well as an exogene variable.
        station (str): The selected weather station.
        max_samples (int): Maximum amount returned samples.
    """
    if dataset == "covid": #Ensure that only data from the same country gets into one sample
        return _get_samples_from_covid_data(n, x_attributes, max_samples, serialize)

    elif dataset == "weather":
        return _get_samples_from_weather_data(n, x_attributes, station, max_samples, serialize)

    else:
        raise Exception(f'{dataset} is an unknown dataset')


def check_separate_weather_data(dataset: str, scenario: str, stations: list, client_num: int):
    if dataset == "weather" and scenario == "separate":
        if client_num != len(stations):
            raise Exception(f'The number of clients is {str(client_num)}, while the number of stations is {str(len(stations))}.\nChange FL_SCENARIO to "mixed" or make sure the number of clients and weather stations is the same.')


def set_initial_parameters(model, shape) -> None:
    model.coef_ = np.zeros(shape)

    try:
        model.intercept_ = 0
    except AttributeError:
        return


def _check_covid_dataset(data: DataFrame) -> bool:
    return (data.location == data.location.iloc[0]).all()


def _check_for_nans(data: DataFrame) -> bool:
    return not data.isnull().values.any()





def _get_samples_from_covid_data(n: int, attributes: list[str], num_of_samples: int, serialize: bool):
    """
    Generates samples from the covid dataset.
    Args:
        n (int): Numbers of records per sample.
        attributes (list[str]): List of attributes that will be used from the dataset. The fist element is the endogene variable.
        num_of_samples (int): Number of returned samples.
    """

    #load data
    data = pd.read_csv(Path(__file__).parent.parent.joinpath("datasets", "horizontal", "covid", "owid-covid-data.csv"))

    #load data if already serialized
    path = Path(__file__).parent.parent.joinpath("datasets", "samples", f'covid_{n}_{"_".join(attributes)}.pkl')
    if path.exists():
        pkl_file = open(path, 'rb')
        x_data, y_data = pickle.load(pkl_file)
        pkl_file.close()
        
        if len(data.index) > num_of_samples:
            return x_data[:num_of_samples], y_data[:num_of_samples]

        else:
            return x_data[:len(data.index) - 1], y_data[:len(data.index) - 1]


    #scale the data
    record_info = data[["iso_code", "continent", "location", "date", "tests_units"]]
    data = data.drop(["iso_code", "continent", "location", "date", "tests_units"], axis=1)
    data_columns = data.columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data_columns)
    data = pd.concat([data, record_info], axis=1)


    x_data = []
    y_data = []

    #split the data
    countries = data.iso_code.drop_duplicates(keep="first")
    countries = countries[countries != "ESH"] #drop ESH because it only has one entry

    for country in countries:
        country_data = data[data.iso_code == country]

        splitter = SlidingWindowSplitter(fh=1, window_length=n)
        samples = splitter.split_series(country_data[attributes].to_numpy())

        for sample in samples:
            x_sample = sample[0].flatten()
            y_sample = sample[1].flatten()[attributes.index("new_cases")]

            if not np.isnan(np.sum(x_sample)) and not np.isnan(y_sample): #check for nans
                    x_data.append(x_sample)
                    y_data.append(y_sample)

    #save data
    if serialize:
        output = open(path, "wb")
        pickle.dump((x_data, y_data), output)
        output.close()
    
    #enusre that sample number is not out of range
    if len(data.index) > num_of_samples:
        return x_data[:num_of_samples], y_data[:num_of_samples]
    else:
        return x_data[:len(data.index) - 1], y_data[:len(data.index) - 1]


def _get_samples_from_weather_data(n: int, attributes: list, station: str, num_of_samples: int, serialize: bool):
    #load data
    dataset_path = Path(__file__).parent.parent.joinpath("datasets", "vertical", "weather", f"{station}.csv")
    data = pd.read_csv(dataset_path, names=["time", "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"])

    #load data if already serialized
    path = Path(__file__).parent.parent.joinpath("datasets", "samples", f'weather_{n}_{station}_{"_".join(attributes)}.pkl')
    if path.exists():
        pkl_file = open(path, 'rb')
        x_data, y_data = pickle.load(pkl_file)
        pkl_file.close()
        
        if len(data.index) > num_of_samples:
            return x_data[:num_of_samples], y_data[:num_of_samples]

        else:
            return x_data[:len(data.index) - 1], y_data[:len(data.index) - 1]


    #scale the data
    data = data.drop("time", axis=1)
    data_columns = data.columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data_columns)

    #split the data
    splitter = SlidingWindowSplitter(fh=1, window_length=n)
    samples = splitter.split_series(data[attributes].to_numpy())

    x_data = []
    y_data = []

    for sample in samples:
        x_sample = sample[0].flatten()
        y_sample = sample[1].flatten()[0] #the endogene temperature variable

        if not np.isnan(np.sum(x_sample)) and not np.isnan(y_sample): #check for nans
            x_data.append(x_sample)
            y_data.append(y_sample)

    #save data
    if serialize:
        output = open(path, "wb")
        pickle.dump((x_data, y_data), output)
        output.close()

    #enusre that sample number is not out of range
    if len(data.index) > num_of_samples:
        return x_data[:num_of_samples], y_data[:num_of_samples]
    else:
        return x_data[:len(data.index) - 1], y_data[:len(data.index) - 1]


def create_client(name: str, X, Y, entries_per_sample: int, x_attributes: list, loss: str, mlp_hidden_layers: int, testing_data_percentage: float) -> fl.client.NumPyClient:
    """
    Create a Flower Client.

    Args:
    name (String): Specify which algorithm you want to use.\n
        "linear regression"\n
        "linearSVR"\n
        "MLP regressor"\n
        "decision tree" (not implemented yet)\n
        "DL" (not implemented yet)\n
    X: Exogene variables for training and testing.
    Y: Endogene variable for training and testing.
    entries_per_sample (int): Number of X values per sample.
    x_attributes (list): List with names of attributes for X values.
    loss (String): Loss function used for evaluation.
    mlp_hidden_layers (int): Number of layers for Multi-layer perceptron.
    testing_data_percentage (float): Percentage of data used for testing. Must be between 0 and 1.
    """
    available_models = ["linear regression", "linearSVR", "MLP regressor", "decision tree", "DL"]
    available_loss_functions = ["MSE", "MAE", "R2"]
    selected_sk_loss = None
    selected_tf_loss = tf.keras.losses.MeanSquaredError
    selected_tf_metric = tf.keras.metrics.MeanSquaredError

    match loss:
        case "MSE":
            selected_sk_loss = mean_squared_error
            selected_tf_loss = tf.keras.losses.MeanSquaredError
            selected_tf_metric = tf.keras.metrics.MeanSquaredError
        case "MAE":
            selected_sk_loss = mean_absolute_error
            selected_tf_loss = tf.keras.losses.MeanAbsoluteError
            selected_tf_metric = tf.keras.losses.MeanAbsoluteError
        case "R2":
            selected_sk_loss = r2_score
        case _:
            raise Exception(f'{loss} is not a supported loss function')


    match name:
        case "linear regression":
            model = LinearRegression()
            set_initial_parameters(model, (entries_per_sample-1) * len(x_attributes))
            return SklearnClient(model, X, Y, selected_sk_loss, testing_data_percentage)

        case "linearSVR":
            model = LinearSVR()
            set_initial_parameters(model, (entries_per_sample-1) * len(x_attributes))
            return SklearnClient(model, X, Y, selected_sk_loss, testing_data_percentage)

        case "MLP regressor":
            input_shape = np.array(X).shape[1]
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(input_shape,)))
            #add hidden layers
            for _ in range(mlp_hidden_layers):
                model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Dense(1))

            model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=selected_tf_loss(),
                metrics=selected_tf_metric())

            return TFClient(model, X, Y, 10, 0.2)


        case _:
            raise Exception(f'{name} is not a supported algorithm')
