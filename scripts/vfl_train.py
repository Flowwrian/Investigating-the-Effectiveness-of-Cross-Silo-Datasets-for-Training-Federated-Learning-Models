from math import floor
import time

import pandas as pd
import tensorflow as tf

from vfl_models import MLPClientModel, MLPServerModel, LSTMCLientModel, CNNClientModel
import helper

def start_vertical_federated_learning_simulation(attributes: list[str], model_type: str, num_of_entries: int, test_percentage: float, num_of_clients=2, client_output_neurons=16, num_of_hidden_layers=2, batch_size=500, max_samples=30000, epochs=10):
    """
    Start vertical federated learning.\n
    Args:\n
        `attributes` (list[str]): List of attributes used from the covid dataset
        `model_type` (str): Selected model architecture; available options: "MLP", "LSTM", "CNN"
        `num_of_entries` (int): Number of entries per sample
        `test_percentage` (float): Percentage of data used for testing
        `num_of_clients` (int): Number of clients
        `client_output_neurons` (int): Number of output neurons of every client
        `num_of_hidden_layers` (int): Number of hidden layers for client models
        `neurons_per_hidden_layer` (int): Number of neurons per hidden layer
        `batch_size` (int): Batch size
        `max_samples` (int): Number of samples used for training; If larger than available records, the whole dataset gets used
        `epochs` (int): Number of epochs
    """
    CLIENTS = num_of_clients
    CLIENT_OUTPUT_NEURONS = client_output_neurons
    MODEL_TYPE = model_type # ["MLP", "LSTM", "CNN"]
    ATTRIBUTES = attributes # ["new_cases", "weekly_hosp_admissions"]
    NUM_OF_ENTRIES = num_of_entries
    TEST_PERCENTAGE = test_percentage
    NUM_OF_HIDDEN_LAYERS = num_of_hidden_layers
    BATCH_SIZE = batch_size
    MAX_SAMPLES = max_samples
    EPOCHS = epochs

    if CLIENTS != len(ATTRIBUTES):
        raise Exception(f'Clients has to be the same as attribute length. {CLIENTS} clients and {len(ATTRIBUTES)} attributes found.')


    #load data
    data = []
    targets = []
    for attribute in ATTRIBUTES:
        if targets == []:
            X, y = helper.get_samples("covid", NUM_OF_ENTRIES, [attribute], "", True, max_samples=MAX_SAMPLES)
            targets = tf.data.Dataset.from_tensor_slices(y).batch(BATCH_SIZE)

            tf_dataset = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
            data.append(tf_dataset)

        else:
            X, _ = helper.get_samples("covid", NUM_OF_ENTRIES, [attribute], "", True, max_samples=MAX_SAMPLES)
            tf_dataset = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
            data.append(tf_dataset)

    #add targets as last entry
    data.append(targets)
    tf_dataset = tf.data.Dataset.zip(tuple(data))

    #create test dataset
    test_dataset_length = floor((MAX_SAMPLES/BATCH_SIZE)*TEST_PERCENTAGE)
    tf_test_dataset = tf_dataset.take(test_dataset_length)
    tf_dataset = tf_dataset.skip(test_dataset_length)
    print(f'Training batches: {len(list(tf_dataset.as_numpy_iterator()))}, Validation batches: {len(list(tf_test_dataset.as_numpy_iterator()))}')


    match MODEL_TYPE:
        case "MLP":
            clients = []
            for _ in range(CLIENTS):
                clients.append(MLPClientModel(NUM_OF_ENTRIES, CLIENT_OUTPUT_NEURONS, (NUM_OF_HIDDEN_LAYERS, 'relu')))
            server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)

        case "LSTM":
            clients = []
            for _ in range(CLIENTS):
                clients.append(LSTMCLientModel(NUM_OF_ENTRIES, CLIENT_OUTPUT_NEURONS, NUM_OF_HIDDEN_LAYERS))
            server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)
        
        case "CNN":
            clients = []
            for _ in range(CLIENTS):
                clients.append(CNNClientModel(NUM_OF_ENTRIES, CLIENT_OUTPUT_NEURONS,(NUM_OF_HIDDEN_LAYERS, 'relu')))
            server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)
        
        case _:
            raise Exception(f'Unkown model type "{MODEL_TYPE}"')


    #build model
    outputs = []
    for i in range(CLIENTS):
        input_data = list(tf_dataset.take(1).as_numpy_iterator())[0][i] #type:ignore
        outputs.append(clients[i](input_data))
    server_input = tf.concat(outputs, axis=-1)
    server_model(server_input)

    #initzialize optimizers
    optimizer_list = []
    for _ in range(CLIENTS + 1):
        optimizer_list.append(tf.keras.optimizers.Adam())

    #add all variables to a list
    all_trainable_variables = []
    for i, model in enumerate(clients):
        all_trainable_variables += model.trainable_variables
        optimizer_list[i].build(model.trainable_variables)
    all_trainable_variables += server_model.trainable_variables
    optimizer_list[-1].build(server_model.trainable_variables)


    loss_fn = tf.keras.losses.MeanAbsoluteError()

    #results are saved in here
    logs = []
    test_losses = []
    timestamps = []


    #start training
    for epoch in range(EPOCHS):
        #forward
        for batch in tf_dataset:
            with tf.GradientTape() as tape:
                outputs = []
                #client output
                for i, model in enumerate(clients):
                    outputs.append(model(batch[i])) #type:ignore

                server_input = tf.concat(outputs, axis=-1)
                server_output = server_model(server_input)
                loss = loss_fn(batch[-1], server_output) #type:ignore

            #compute gradients
            gradients = tape.gradient(loss, all_trainable_variables)

            #apply gradients
            start_idx = 0
            for i, model in enumerate(clients):
                end_idx = start_idx + len(model.trainable_variables)
                optimizer_list[i].apply_gradients(zip(gradients[start_idx:end_idx], model.trainable_variables)) #type:ignore
                start_idx = end_idx
            #apply gradient to server model
            optimizer_list[-1].apply_gradients(zip(gradients[start_idx:len(server_model.trainable_variables)], server_model.trainable_variables)) #type:ignore

        #calculate loss on test dataset
        test_loss = 0
        for test_batch in tf_test_dataset:
            outputs = []
            for i, model in enumerate(clients):
                outputs.append(model(test_batch[i])) #type:ignore
            server_input = tf.concat(outputs, axis=-1)
            server_output = server_model(server_input)
            test_loss += loss_fn(test_batch[-1], server_output).numpy() #type:ignore
        test_loss = test_loss / test_dataset_length


        print(f'#{epoch} loss: {loss} | gradients: {tf.reduce_sum(gradients[0])} | test loss: {test_loss}') #type:ignore
        logs.append([epoch, loss.numpy(), tf.reduce_sum(gradients[0]).numpy()]) #type:ignore
        test_losses.append((epoch+1, test_loss)) #type:ignore

        timestamps.append((epoch+1,time.perf_counter()))

    df = pd.DataFrame(logs, columns=["epoch", "loss", "gradient"])
    df["time"] = timestamps
    print(df.head())
    #return losses (must be in the same shape as flower history object for serialization)
    return {"losses_distributed": test_losses, "time": timestamps}