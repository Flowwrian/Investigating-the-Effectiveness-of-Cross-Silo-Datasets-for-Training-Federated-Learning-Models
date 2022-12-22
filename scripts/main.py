import math

import pandas as pd
import flwr

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import helper
from helper import Library
from sklearn_client import SklearnClient
from tf_client import TFClient


#Set your parameters here
#Dataset
DATA_PATH = "/home/florian/bachelorarbeit/code/Cross-Silo-FL/datasets/horizontal/covid/owid-covid-data.csv"
DATASET = helper.Dataset.Covid
ENTRIES_PER_SAMPLE = 10
NUMBER_OF_SAMPLES = 100
X_ATTRIBUTES = ["total_cases", "new_cases"]
Y_ATTRIBUTE = "new_cases"
PERCENTAGE_OF_TESTING_DATA = 0.2
#Clients
NUMBER_OF_CLIENTS = 5
#Model
LIBRARY = Library.Sklearn
MODEL = LinearRegression
#Scikit-Learn options
LOSS = mean_squared_error
#Tensorflow options
EPOCHS = 10

#Misc
VERBOSE = True


if __name__ == "__main__":
    #preprocess data
    data = pd.read_csv(DATA_PATH)
    x_train, y_train = helper.get_samples(data, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, Y_ATTRIBUTE, DATASET, NUMBER_OF_SAMPLES)
    #x_train, x_test, y_train, y_test = helper.sample_split(data_samples, PERCENTAGE_OF_TESTING_DATA, X_ATTRIBUTES, Y_ATTRIBUTE)
    if VERBOSE:
        print(f'Data loaded from {DATA_PATH} \nSplit into {str(NUMBER_OF_SAMPLES)} samples')

    #define client_fn (here it's easier to access the dataset)
    def client_fn(cid: str) -> flwr.client.NumPyClient:
        #sample splitting logic from https://github.com/adap/flower/blob/b0bb1bb990373c35feaca9aca37c790fed029cf9/examples/simulation_tensorflow/sim.py#L48
        partition_size = math.floor(len(x_train) / NUMBER_OF_CLIENTS)
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

        x_train_cid = x_train[idx_from:idx_to]
        y_train_cid = y_train[idx_from:idx_to]

        if VERBOSE:
            print(f'Client {cid} starting...')
        if LIBRARY == Library.Sklearn:
            #instantiate new model
            new_model = MODEL()
            #setting inital parameters for model
            helper.set_initial_parameters(new_model, LIBRARY, (ENTRIES_PER_SAMPLE-1) * len(X_ATTRIBUTES))

            return SklearnClient(new_model, x_train_cid, y_train_cid, LOSS, PERCENTAGE_OF_TESTING_DATA)
        else:
            return TFClient(MODEL, x_train_cid, y_train_cid, x_test_cid, y_test_cid, EPOCHS)
        

    #start simulation
    hist = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUMBER_OF_CLIENTS,
        config = flwr.server.ServerConfig()
    )