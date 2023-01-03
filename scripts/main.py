import math
import time

import flwr

import helper


#Set your parameters here
#Dataset
DATA = "weather"
ENTRIES_PER_SAMPLE = 10
NUMBER_OF_SAMPLES = 1000000
X_ATTRIBUTES = ["temp", "pres", "tsun"]
Y_ATTRIBUTE = "temp"
#Weather station details
STATIONS = ["muenchen"]
FL_SCENARIO = "mixed" # "mixed", "separate"
PERCENTAGE_OF_TESTING_DATA = 0.2
ROUNDS = 1
#Clients
NUMBER_OF_CLIENTS = 5
#Model
MODEL = "MLP regressor"
LOSS = "MSE"
#Tensorflow options
EPOCHS = 10

#Misc
VERBOSE = True


if __name__ == "__main__":
    start_sampling_timer = time.time()
    #preprocess data
    x_train, y_train = helper.get_samples(DATA, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, Y_ATTRIBUTE, STATIONS[0], NUMBER_OF_SAMPLES) # type: ignore 
    if VERBOSE:
        print(f'Data loaded after {time.time() - start_sampling_timer}')
        print(f'{DATA} data loaded\nSplit into {str(NUMBER_OF_SAMPLES)} samples')

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
    start_simulation_timer = time.time()
    hist = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUMBER_OF_CLIENTS,
        config = flwr.server.ServerConfig(num_rounds=ROUNDS)
    )
    print(f'Training finished after {time.time() - start_simulation_timer}')