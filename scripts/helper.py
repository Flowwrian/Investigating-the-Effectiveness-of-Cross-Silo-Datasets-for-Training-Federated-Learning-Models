from pathlib import Path
import pickle
from time import strftime, localtime
import argparse

import flwr as fl
from flwr.server import History
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sktime.forecasting.model_selection import SlidingWindowSplitter

from sklearn_client import SklearnClient
from tf_client import TFClient


def get_samples(dataset: str, n: int, x_attributes: list[str], station: str, serialize: bool, max_samples: int = 100000, standardize = False):
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
        return _get_samples_from_covid_data(n, x_attributes, max_samples, serialize, standardize)

    elif dataset == "weather":
        return _get_samples_from_weather_data(n, x_attributes, station, max_samples, serialize, standardize)

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





def _get_samples_from_covid_data(n: int, attributes: list[str], num_of_samples: int, serialize: bool, standardize=True):
    """
    Generates samples from the covid dataset.
    Args:
        n (int): Numbers of records per sample.
        attributes (list[str]): List of attributes that will be used from the dataset. The fist element is the endogene variable.
        num_of_samples (int): Number of returned samples.
    """

    #load data
    data = pd.read_csv(Path(__file__).parent.parent.joinpath("datasets", "vertical", "covid", "owid-covid-data.csv"))

    #load data if already serialized
    path = Path(__file__).parent.parent.joinpath("datasets", "samples", f'covid_{n}n_{num_of_samples}samples_{"_".join(attributes)}_{standardize}scaled.pkl')
    if path.exists():
        pkl_file = open(path, 'rb')
        X, y = pickle.load(pkl_file)
        pkl_file.close()
        
        return X, y 

    #fill nan with 0
    data[attributes] = data[attributes].fillna(0)
    data["new_cases"] = data["new_cases"].fillna(0)

    # #scale the data
    record_info = data[["iso_code", "continent", "location", "date", "tests_units"]]

    #scale selected attributes
    if standardize:
        selected_data = data[attributes]
        selected_data_columns = selected_data.columns
        scaler = StandardScaler()
        scaled_selected_data = scaler.fit_transform(selected_data)
        selected_data = pd.DataFrame(scaled_selected_data, columns=selected_data_columns)
        data = pd.concat([selected_data, record_info], axis=1)


    X = np.empty(shape=(num_of_samples, n*len(attributes)))
    y = np.empty(shape=(num_of_samples,))

    countries = data.iso_code.drop_duplicates(keep="first")
    countries = countries[countries != "ESH"] #drop ESH because it only has one entry

    i = 0
    for country in countries:
        country_data = data[data.iso_code == country]

        splitter = SlidingWindowSplitter(fh=1, window_length=n)
        samples = splitter.split_series(country_data[attributes].to_numpy())

        for features, labels in samples:
            X[i] = features.flatten()
            y[i] = labels.flatten()[0]

            i += 1

            if i == num_of_samples:
                break

        if i == num_of_samples:
            break

    #save data
    if serialize:
            output = open(path, "wb")
            pickle.dump((X, y), output)
            output.close()

    return X, y



def _get_samples_from_weather_data(n: int, attributes: list, station: str, num_of_samples: int, serialize: bool, standardize = False):

    #load data if already serialized
    path = Path(__file__).parent.parent.joinpath("datasets", "samples", f'weather_{n}n_{num_of_samples}samples_{station}_{"_".join(attributes)}_{standardize}scaled.pkl')
    if path.exists():
        pkl_file = open(path, 'rb')
        X, y = pickle.load(pkl_file)
        pkl_file.close()
        
        return X, y 
    
    
    
    #load data
    data = pd.read_csv(Path(__file__).parent.parent.joinpath("datasets", "horizontal", "weather", f"{station}.csv"), names=["time", "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"])


    #scale the data
    if standardize:
        data = data.drop("time", axis=1)
        data_columns = data.columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        data = pd.DataFrame(scaled_data, columns=data_columns)

    data = data[attributes]
    data = data.dropna()
    #data = data.fillna(method="pad")

    X = np.empty(shape=(num_of_samples, n*len(attributes)))
    y = np.empty(shape=(num_of_samples,))


    #split the data
    splitter = SlidingWindowSplitter(fh=1, window_length=n)
    samples = splitter.split_series(data[attributes].to_numpy())

    i = 0
    for features, label in samples:
        X[i] = features.flatten()
        y[i] = label.flatten()[0]

        i += 1

        if i == num_of_samples:
            break

    #save data
    if serialize:
            output = open(path, "wb")
            pickle.dump((X, y), output)
            output.close()

    return X, y


def create_client(name: str, X, Y, entries_per_sample: int, x_attributes: list, loss: str, hidden_layers: int, epochs: int, batch_size: int, testing_data_percentage: float) -> fl.client.NumPyClient:
    """
    Create a Flower Client.

    Args:
    name (String): Specify which algorithm you want to use.\n
        "linear regression"\n
        "linearSVR"\n
        "MLP"\n
        "decision tree" (not implemented yet)\n
        "DL" (not implemented yet)\n
    X: Exogene variables for training and testing.
    Y: Endogene variable for training and testing.
    entries_per_sample (int): Number of X values per sample.
    x_attributes (list): List with names of attributes for X values.
    loss (String): Loss function used for evaluation.
    hidden_layers (int): Number of hidden layers for Tensorflow models.
    epochs (int): Number of epochs every client trains before sending parameters to the server.
    batch_size (int): Number of entries per batch. Only relevant for Tensorflow models.
    testing_data_percentage (float): Percentage of data used for testing. Must be between 0 and 1.
    """
    available_models = ["linear regression", "linearSVR", "MLP", "decision tree", "DL"]
    available_loss_functions = ["MSE", "MAE", "R2", "MAPE"]
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
            selected_tf_metric = tf.keras.metrics.MeanAbsoluteError
        case "MAPE":
            selected_sk_loss = mean_absolute_percentage_error
            selected_tf_loss = tf.keras.losses.MeanAbsolutePercentageError
            selected_tf_metric = tf.keras.metrics.MeanAbsolutePercentageError
        case "R2":
            selected_sk_loss = r2_score
        case _:
            raise Exception(f'{loss} is not a supported loss function')


    match name:
        case "linear regression":
            model = LinearRegression(fit_intercept=False, positive=True)
            set_initial_parameters(model, (entries_per_sample-1) * len(x_attributes))
            return SklearnClient(model, X, Y, selected_sk_loss, testing_data_percentage)

        case "linearSVR":
            model = LinearSVR(C=0.1, epsilon=0, tol=0.00001)
            set_initial_parameters(model, (entries_per_sample-1) * len(x_attributes))
            return SklearnClient(model, X, Y, selected_sk_loss, testing_data_percentage)

        case "MLP":
            optimal_parameters = [328, 488, 432, 136, 288]

            input_shape = np.array(X).shape[1]
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(input_shape,)))
            #add hidden layers
            for i in range(hidden_layers):
                try:
                    model.add(tf.keras.layers.Dense(optimal_parameters[i], activation="relu"))
                except:
                    model.add(tf.keras.layers.Dense(32, activation="relu"))
            model.add(tf.keras.layers.Dense(1, activation="linear"))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=selected_tf_loss(),
                metrics=selected_tf_metric())

            return TFClient(model, X, Y, epochs, batch_size, testing_data_percentage)

        case "LSTM":
            input_shape = np.array(X).shape[1]
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Reshape((int(input_shape/len(x_attributes)), len(x_attributes)), input_shape=(input_shape,)))
            for _ in range(hidden_layers - 1):
                model.add(tf.keras.layers.LSTM(96, return_sequences=True))
            model.add(tf.keras.layers.LSTM(64))
            model.add(tf.keras.layers.Dense(24, activation="relu"))
            model.add(tf.keras.layers.Dense(16, activation="relu"))
            model.add(tf.keras.layers.Dense(1, activation="linear"))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=selected_tf_loss(),
                metrics=selected_tf_metric())

            return TFClient(model, X, Y, epochs, batch_size, testing_data_percentage)

        case "CNN":
            optimal_filter = [64, 96, 96, 96]
            optimal_kernel = [3, 3, 4, 3]


            input_shape = np.array(X).shape[1]
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Reshape((int(input_shape/len(x_attributes)), len(x_attributes)), input_shape=(input_shape,)))
            for i in range(hidden_layers):
                try:
                    model.add(tf.keras.layers.Conv1D(optimal_filter[i], optimal_kernel[i], padding="same", activation="relu"))
                except:
                    model.add(tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"))
            model.add(tf.keras.layers.Dense(112, activation="relu"))
            model.add(tf.keras.layers.Dense(104, activation="relu"))
            model.add(tf.keras.layers.Dense(1, activation="linear"))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss= selected_tf_loss(),
                metrics= selected_tf_metric()
            )

            return TFClient(model, X, Y, epochs, batch_size, testing_data_percentage)

        case _:
            raise Exception(f'{name} is not a supported algorithm')


def save_results(history, args: argparse.Namespace):
    """
    Save the results of training to logs.json.
    Args:
        history (flwr.server.History): History object returned from simulation.
        args (argparse.Namespace):  Selected training options.
    """
    columns = ["date", "model", "dataset", "rounds", "losses_distributed", "time", "number_of_clients", "entries", "number_of_samples", "attributes", "stations", "scenario", "percentage_of_testing_data", "loss", "epochs", "hidden_layers"]
    date = strftime("%Y-%m-%d %H:%M:%S", localtime())
    try: #results if returned from flower simulation
        results = pd.DataFrame([[date, args.model, args.dataset, args.rounds, history.losses_distributed, history.metrics_distributed["time"], args.clients, args.entries, args.samples, args.attributes, args.stations, args.scenario, args.testing_data, args.loss, args.epochs, args.hidden_layers]], columns=columns)
    except: #results if returned from vertical FL
        results = pd.DataFrame([[date, args.model, args.dataset, args.rounds, history["losses_distributed"], history["time"], args.clients, args.entries, args.samples, args.attributes, args.stations, args.scenario, args.testing_data, args.loss, args.epochs, args.hidden_layers]], columns=columns)

    #read in json file and append new data
    try:
        old_data = pd.read_json(Path(__file__).parent.parent.joinpath("logs.json"))
        pd.concat([old_data, results], ignore_index=True).to_json(Path(__file__).parent.parent.joinpath("logs.json"))
        print("Results saved to logs.json.")
    except:
        results.to_json(Path(__file__).parent.parent.joinpath("logs.json"))
        print("New logs.json file created. Results saved.")

def training_time(metrics):
    #return the timestamp of the last trained client
    times = [m["time"] for i, m in metrics]
    return {"time": times[-1]}