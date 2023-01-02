import math

import pandas as pd
import flwr

import helper


#Set your parameters here
#Dataset
DATA_PATH = "/home/florian/bachelorarbeit/code/Cross-Silo-FL/datasets/horizontal/covid/owid-covid-data.csv"
DATASET = helper.Dataset.Covid
ENTRIES_PER_SAMPLE = 10
NUMBER_OF_SAMPLES = 100
X_ATTRIBUTES = ["total_cases", "new_cases"]
Y_ATTRIBUTE = "new_cases"
PERCENTAGE_OF_TESTING_DATA = 0.2
ROUNDS = 1
#Clients
NUMBER_OF_CLIENTS = 5
#Model
MODEL = "MLP regressor"
#Scikit-Learn options
LOSS = "MSE"
#Tensorflow options
EPOCHS = 10

#Misc
VERBOSE = True


if __name__ == "__main__":
    #preprocess data
    data = pd.read_csv(DATA_PATH)
    x_train, y_train = helper.get_samples(data, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, Y_ATTRIBUTE, DATASET, NUMBER_OF_SAMPLES)
    if VERBOSE:
        print(f'Data loaded from {DATA_PATH} \nSplit into {str(NUMBER_OF_SAMPLES)} samples')

    #define client_fn (here it's easier to access the dataset)
    def client_fn(cid: str) -> flwr.client.NumPyClient:
        #sample splitting logic from https://github.com/adap/flower/blob/b0bb1bb990373c35feaca9aca37c790fed029cf9/examples/simulation_tensorflow/sim.py#L48
        partition_size = math.floor(len(x_train) / NUMBER_OF_CLIENTS)
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

        x_train_cid = x_train[idx_from:idx_to]
        y_train_cid = y_train[idx_from:idx_to]
       
        client = helper.create_client(MODEL, x_train_cid, y_train_cid, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, LOSS, PERCENTAGE_OF_TESTING_DATA)
        return client


    #start simulation
    hist = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUMBER_OF_CLIENTS,
        config = flwr.server.ServerConfig(num_rounds=ROUNDS)
    )