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
STATIONS = ["muenchen", "potsdam"]
FL_SCENARIO = "separate" # "mixed", "separate"
PERCENTAGE_OF_TESTING_DATA = 0.2
ROUNDS = 1
#Clients
NUMBER_OF_CLIENTS = 2
#Model
MODEL = "MLP regressor"
LOSS = "MSE"
#Tensorflow options
EPOCHS = 10
MLP_HIDDEN_LAYERS = 1

#Misc
VERBOSE = True


if __name__ == "__main__":
    #check if weather number of stations equals number of clients; only important for DATA = "weather" and FL_SCENARIO = "separate"
    helper.check_separate_weather_data(DATA, FL_SCENARIO, STATIONS, NUMBER_OF_CLIENTS)

    start_sampling_timer = time.time()
    #preprocess data
    if DATA == "covid":
        x_train, y_train = helper.get_samples(DATA, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, Y_ATTRIBUTE, STATIONS[0], NUMBER_OF_SAMPLES) # type: ignore 

    if DATA == "weather" and FL_SCENARIO == "mixed":
        samples_per_station = math.floor(NUMBER_OF_SAMPLES/len(STATIONS))
        x_train = []
        y_train = []

        #combine all stations data
        for station in STATIONS:
            new_x_train, new_y_train = helper.get_samples(DATA, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, Y_ATTRIBUTE, STATIONS[0], samples_per_station)
            x_train = x_train + new_x_train
            y_train = y_train + new_x_train

    if VERBOSE:
        print(f'Data loaded after {time.time() - start_sampling_timer}')
        print(f'{DATA} data loaded\nSplit into {str(NUMBER_OF_SAMPLES)} samples')


    def client_fn(cid: str) -> flwr.client.NumPyClient:
        #create data for separate weather test case
        if DATA == "weather" and FL_SCENARIO == "separate":
            samples_per_station = math.floor(NUMBER_OF_SAMPLES/len(STATIONS))
            x_train_cid, y_train_cid = helper.get_samples(DATA, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, Y_ATTRIBUTE, STATIONS[int(cid)], samples_per_station)
        

        #all other test cases
        else:
            #sample splitting logic from https://github.com/adap/flower/blob/b0bb1bb990373c35feaca9aca37c790fed029cf9/examples/simulation_tensorflow/sim.py#L48
            partition_size = math.floor(len(x_train) / NUMBER_OF_CLIENTS)
            idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

            x_train_cid = x_train[idx_from:idx_to]
            y_train_cid = y_train[idx_from:idx_to]
       
        client = helper.create_client(MODEL, x_train_cid, y_train_cid, ENTRIES_PER_SAMPLE, X_ATTRIBUTES, LOSS, MLP_HIDDEN_LAYERS, PERCENTAGE_OF_TESTING_DATA)
        return client


    #start simulation
    start_simulation_timer = time.time()
    hist = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUMBER_OF_CLIENTS,
        config = flwr.server.ServerConfig(num_rounds=ROUNDS)
    )
    print(f'Training finished after {time.time() - start_simulation_timer}')