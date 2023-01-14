import numpy as np
import pandas as pd
from pandas import DataFrame
import flwr as fl

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_selection import SlidingWindowSplitter

import tensorflow as tf

from sklearn_client import SklearnClient
from tf_client import TFClient


def get_samples(dataset: str, n: int, x_attributes: list[str], y_attribute: str, station: str, max_samples: int = 100000):
    """
    Generate samples from a given dataset. The samples are created by using a rolling window.
    Args:
        data (DataFrame): The dataset sampling from.
        n (int): Number of records per sample.
        x_attributes (list[str]): The exogene variables used for input.
        y_attribute (str): The endogene variable (the expected model output).
        dataset (Dataset): Enum for specifing which dataset you want to sample from.
            Dataset.Covid: owid-covid-data.csv
            Dataset.Weather: Weather data from ___
            Dataset.Other: other datasets
        max_samples (int): Maximum amount returned samples.
    """
    if dataset == "covid": #Ensure that only data from the same country gets into one sample
        return _get_samples_from_covid_data(n, x_attributes, y_attribute, max_samples)

    elif dataset == "weather":
        return _get_samples_from_weather_data(n, x_attributes, y_attribute, station, max_samples)

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





def _get_samples_from_covid_data(n: int, x_attributes: list[str], y_attribute: str, max_samples):
    data = pd.read_csv("/home/florian/bachelorarbeit/code/Cross-Silo-FL/datasets/horizontal/covid/owid-covid-data.csv")
    samples = list()
    num_of_rows = len(data.index)

    #get the relevant rows & scale them
    data_locations = data.location
    scaler = StandardScaler()
    if y_attribute in x_attributes:
        scaled_data = scaler.fit_transform(data[x_attributes])
        data = pd.DataFrame(scaled_data, columns=x_attributes)
    else:
        scaled_data = scaler.fit_transform(data[[*x_attributes, y_attribute]])
        data = pd.DataFrame(scaled_data, columns=[*x_attributes, y_attribute])
    data["location"] = data_locations


    for i in range(num_of_rows):
        if i+n > num_of_rows or len(samples) == max_samples:
            break
        
        #check for nans
        if y_attribute in x_attributes:
            new_data = data.iloc[range(i,i+n)] # type: ignore
        else:
            new_data = data.iloc[range(i,i+n)] # type: ignore

        if _check_covid_dataset(data.iloc[range(i,i+n)]) and _check_for_nans(new_data): # type: ignore
            samples.append(new_data)

    data_series = pd.Series(samples)

    #logic from sample_split starts here
    sample_length = len(data_series.iloc[0].index)
    x_data = []
    y_data = []

    for i in range(len(data_series)):
        current_sample = data_series.iloc[i]
        new_x_data = current_sample.iloc[range(0, sample_length - 1)][x_attributes]
        new_y_data = current_sample.iloc[sample_length - 1][y_attribute]
        
        #if _check_for_nans(new_x_data) and _check_for_nans(new_y_data):
        x_data.append(new_x_data.to_numpy().flatten())
        y_data.append(new_y_data)

    return x_data, y_data


def _get_samples_from_weather_data(n: int, x_attributes: list, y_attribute: str, station: str, num_of_samples: int):
    #load data
    path = f"/home/florian/bachelorarbeit/code/Cross-Silo-FL/datasets/vertical/weather/{station}.csv"
    data = pd.read_csv(path, names=["time", "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"])
    
    #scale the data
    data = data.drop("time", axis=1)
    data_columns = data.columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data_columns)

    #split the data
    splitter = SlidingWindowSplitter(fh=1, window_length=n)
    samples = splitter.split_series(data[x_attributes].to_numpy())

    x_data = []
    y_data = []

    for sample in samples:
        x_sample = sample[0].flatten()
        y_sample = sample[1].flatten()[0] #the endogene temperature variable

        if not np.isnan(np.sum(x_sample)) and not np.isnan(y_sample): #check for nans
            x_data.append(x_sample)
            y_data.append(y_sample)

    return x_data[:num_of_samples], y_data[:num_of_samples]


def create_client(name: str, x_train, y_train, entries_per_sample: int, x_attributes: list, loss: str, mlp_hidden_layers: int, testing_data_percentage: float) -> fl.client.NumPyClient:
    """
    Create a Flower Client. 

    Args:
    name (String): Specify which algorithm you want to use.\n
        "linear regression"\n
        "linearSVR"\n
        "MLP regressor"\n
        "decision tree"\n
        "DL"
    x_train: Exogene variables for training and testing.
    y_train: Endogene variable for training and testing.
    entries_per_sample (int): Number of X values per sample.
    x_attributes (list): List with names of attributes for X values.
    loss (String): Loss function used for evaluation.
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
            return SklearnClient(model, x_train, y_train, selected_sk_loss, testing_data_percentage)

        case "linearSVR":
            model = LinearSVR()
            set_initial_parameters(model, (entries_per_sample-1) * len(x_attributes))
            return SklearnClient(model, x_train, y_train, selected_sk_loss, testing_data_percentage)

        case "MLP regressor":
            # model = MLPRegressor()
            # model.fit([x_train[0]], [y_train[0]]) #fitting the first sample to generate coefs_
            # return SklearnMLPClient(model, x_train, y_train, selected_loss, testing_data_percentage)
            input_shape = np.array(x_train).shape[1]
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(input_shape,)))
            #add hidden layers
            for layer in range(mlp_hidden_layers):
                model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Dense(1))

            model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=selected_tf_loss(),
                metrics=selected_tf_metric())

            return TFClient(model, x_train, y_train, 10, 0.2)


        case _:
            raise Exception(f'{name} is not a supported algorithm')
