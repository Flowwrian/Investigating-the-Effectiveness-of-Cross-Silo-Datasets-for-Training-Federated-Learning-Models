import argparse
import math
import os
import time

import flwr

import helper
import vfl_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(description="A sandbox FL environment")

    parser.add_argument("--model", "--m", type=str, required=True, choices=[
                        "linear regression", "linearSVR", "MLP", "LSTM", "CNN"], help="ML algorithm used for trining")
    parser.add_argument("--dataset", "--d", type=str, required=True,
                        choices=["covid", "weather"], help="dataset used for training")
    parser.add_argument("--attributes", "--a", nargs="+",
                        required=True, help="selected attributes from the dataset")
    parser.add_argument("--rounds", "--r", type=int,
                        required=True, help="rounds of federated learning")
    parser.add_argument("--clients", "--c", type=int,
                        required=True, help="number of clients")
    parser.add_argument("--loss", "--l", type=str, required=True, choices=[
                        "MSE", "MAE", "R2", "MAPE"], help="selected loss function (R2 only available for Scikit-learn models)")
    parser.add_argument("--stations", "--st", nargs="+", choices=["berlin_alexanderplatz", "frankfurt_am_main_westend", "hamburg_airport",
                        "leipzig", "muenchen", "potsdam"], default=["berlin"], help="OPTIONAL selected weather stations; only relevant for 'weather' dataset")
    parser.add_argument("--entries", "--e", type=int, default=10,
                        help="OPTIONAL number of past days values to predict the next days target value")
    parser.add_argument("--samples", "--sa", type=int,
                        default=100000, help="OPTIONAL number of records to use")
    parser.add_argument("--scenario", "--sc", type=str, choices=["separate", "mixed"], default="separate",
                        help="OPTIONAL scenario of training. Can either be an federated learning or distributed learning setting")
    parser.add_argument("--testing_data", "--t", type=float, default=0.2,
                        help="OPTIONAL percentage of data used for training")
    parser.add_argument("--epochs", "--ep", type=int, default=10,
                        help="OPTIONAL number of epochs; only relevant for Tensorflow models")
    parser.add_argument("--hidden_layers", "--hl", type=int, default=1,
                        help="OPTIONAL number of hidden layers; only relevant for Tensorflow models")
    parser.add_argument("--verbosity", "--v", type=bool,
                        default=True, help="OPTIONAL verbosity of the output")
    parser.add_argument("--serialize", "--se", type=bool, default=True,
                        help="OPTIONAL serialize the sampled data. Drastically reduces time preprocessing the data")
    parser.add_argument("--log", type=bool, default=False,
                        help="OPTIONAL save the results of the training to logs.json")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # PARAMETERS
    # Dataset
    DATA = args.dataset  # "covid", "weather"
    ENTRIES_PER_SAMPLE = args.entries
    NUMBER_OF_SAMPLES = args.samples
    ATTRIBUTES = args.attributes
    # Weather station details
    STATIONS = args.stations
    FL_SCENARIO = args.scenario  # "mixed", "separate"
    PERCENTAGE_OF_TESTING_DATA = args.testing_data
    ROUNDS = args.rounds
    # Clients
    NUMBER_OF_CLIENTS = args.clients
    # Model
    # "linear regression", "linearSVR", "MLP regressor", ("decision tree", "DL" not implemented yet)
    MODEL = args.model
    # "MSE", "MAE", "R2" (R2 only works for scikit-learn models)
    LOSS = args.loss
    # Tensorflow options
    EPOCHS = args.epochs
    MLP_HIDDEN_LAYERS = args.hidden_layers

    # Misc
    VERBOSE = args.verbosity
    SERIALIZE = args.serialize
    LOG = args.log

    # check if weather number of stations equals number of clients; only important for DATA = "weather" and FL_SCENARIO = "separate"
    helper.check_separate_weather_data(
        DATA, FL_SCENARIO, STATIONS, NUMBER_OF_CLIENTS)

    start_sampling_timer = time.time()
    # preprocess data
    if DATA == "weather" and FL_SCENARIO == "mixed":
        samples_per_station = math.floor(NUMBER_OF_SAMPLES/len(STATIONS))
        x_train = []
        y_train = []

        # combine all stations data
        for station in STATIONS:
            new_x_train, new_y_train = helper.get_samples(
                DATA, ENTRIES_PER_SAMPLE, ATTRIBUTES, station, SERIALIZE, samples_per_station)
            x_train = x_train + new_x_train
            y_train = y_train + new_x_train

    if VERBOSE:
        print(f'Data loaded after {time.time() - start_sampling_timer}')
        print(f'{DATA} data loaded\nSplit into {NUMBER_OF_SAMPLES} samples')

    def client_fn(cid: str) -> flwr.client.NumPyClient:
        # create data for separate weather test case
        if DATA == "weather" and FL_SCENARIO == "separate":
            samples_per_station = math.floor(NUMBER_OF_SAMPLES/len(STATIONS))
            x_train_cid, y_train_cid = helper.get_samples(
                DATA, ENTRIES_PER_SAMPLE, ATTRIBUTES, STATIONS[int(cid)], SERIALIZE, samples_per_station)

        # all other test cases
        else:
            partition_size = math.floor(len(x_train) / NUMBER_OF_CLIENTS)
            idx_from, idx_to = int(
                cid) * partition_size, (int(cid) + 1) * partition_size

            x_train_cid = x_train[idx_from:idx_to]
            y_train_cid = y_train[idx_from:idx_to]

        client = helper.create_client(MODEL, x_train_cid, y_train_cid, ENTRIES_PER_SAMPLE,
                                      ATTRIBUTES, LOSS, MLP_HIDDEN_LAYERS, PERCENTAGE_OF_TESTING_DATA)
        return client

    # start simulation
    if DATA == "weather":
        start_simulation_timer = time.time()
        hist = flwr.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUMBER_OF_CLIENTS,
            config=flwr.server.ServerConfig(num_rounds=ROUNDS)
        )
        print(
            f'Training finished after {time.time() - start_simulation_timer}')

        if LOG:
            helper.save_results(hist, args)

    elif DATA == "covid":
        hist = vfl_train.start_vertical_federated_learning_simulation(
            attributes=ATTRIBUTES,
            model_type=MODEL,
            num_of_entries=ENTRIES_PER_SAMPLE,
            num_of_clients=NUMBER_OF_CLIENTS,
            num_of_hidden_layers=MLP_HIDDEN_LAYERS,
            max_samples=NUMBER_OF_SAMPLES,
            epochs=ROUNDS
        )
        if LOG:
            helper.save_results(hist, args)

    else:
        raise Exception(f'Unknown dataset "{DATA}".')
