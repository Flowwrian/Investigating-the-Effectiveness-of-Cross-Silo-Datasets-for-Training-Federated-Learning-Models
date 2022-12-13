import math

import pandas as pd
import flwr

from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss

import helper
from helper import Library
from sklearn_client import SklearnClient
from tf_client import TFClient


#Set your parameters here
#Dataset
DATA_PATH = "C:\\Users\\floha\\Documents\\Bachelorarbeit\\code\\datasets\\horizontal\\covid\\owid-covid-data.csv" #Change backslash to slash if not on Windows
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
MODEL = LinearRegression()
#Scikit-Learn options
LOSS = log_loss
#Tensorflow options
EPOCHS = 10

#Misc
VERBOSE = True


if __name__ == "__main__":
    #preprocess data
    data = pd.read_csv(DATA_PATH)
    data_samples = helper.get_samples(data, ENTRIES_PER_SAMPLE, DATASET, NUMBER_OF_SAMPLES)
    x_train, x_test, y_train, y_test = helper.sample_split(data_samples, PERCENTAGE_OF_TESTING_DATA, X_ATTRIBUTES, Y_ATTRIBUTE)
    if VERBOSE:
        print(f'Data loaded from {DATA_PATH} \nSplit into {str(NUMBER_OF_SAMPLES)} samples')

    #define client_fn (here it's easier to access the dataset)
    def client_fn(cid: str) -> flwr.client.NumPyClient:
        #sample splitting logic from https://github.com/adap/flower/blob/b0bb1bb990373c35feaca9aca37c790fed029cf9/examples/simulation_tensorflow/sim.py#L48
        partition_size = math.floor(len(data_samples) / NUMBER_OF_CLIENTS)
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

        x_train_cid = x_train[idx_from:idx_to]
        y_train_cid = y_train[idx_from:idx_to]

        x_test_cid = x_test[idx_from:idx_to]
        y_test_cid = y_test[idx_from:idx_to]

        if VERBOSE:
            print(f'Client {cid} starting...')
        if LIBRARY == Library.Sklearn:
            return SklearnClient(MODEL, x_train_cid, y_train_cid, x_test_cid, y_test_cid, LOSS)
        else:
            return TFClient(MODEL, x_train_cid, y_train_cid, x_test_cid, y_test_cid, EPOCHS)
        


    #start simulation
    hist = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUMBER_OF_CLIENTS
    )

    print(hist)