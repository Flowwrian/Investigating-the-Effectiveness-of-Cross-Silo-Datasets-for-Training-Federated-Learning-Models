import math

import pandas as pd
import flwr

import helper


#Set your parameters here
#Dataset
DATA_PATH = "C:\\Users\\floha\\Documents\\Bachelorarbeit\\code\\datasets\\horizontal\\covid\\owid-covid-data.csv" #Change backslash to slash if not on Windows
DATASET = helper.Dataset.Covid
ENTRIES_PER_SAMPLE = 10
NUMBER_OF_SAMPLES = 100
X_ATTRIBUTES = [""]
Y_ATTRIBUTE = ""
PERCENTAGE_OF_TESTING_DATA = 0.2
#Clients
NUMBER_OF_CLIENTS = 5

#Details
ATTRIBUTES = []
EPOCHS = 10
VERBOSE = True


if __name__ == "__main__":
    #Preprocess data
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

        #TODO return client
        pass


    #start simulation
    hist = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUMBER_OF_CLIENTS
    )