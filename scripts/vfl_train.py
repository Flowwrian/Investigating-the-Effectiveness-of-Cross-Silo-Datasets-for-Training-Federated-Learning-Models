import pandas as pd
import tensorflow as tf

from vfl_models import MLPClientModel, MLPServerModel, LSTMCLientModel, CNNClientModel
import helper

CLIENTS = 2
CLIENT_OUTPUT_NEURONS = 16
MODEL_TYPE = "CNN" # ["MLP", "LSTM", "CNN"]
ATTRIBUTES = ["new_cases", "weekly_hosp_admissions"]
NUM_OF_SAMPLES = 10
NUM_OF_HIDDEN_LAYERS = 2
NUM_HIDDEN_LAYERS_NEURONS = 32
BATCH_SIZE = 500
MAX_SAMPLES = 30000
EPOCHS = 10

if CLIENTS != len(ATTRIBUTES):
    raise Exception(f'Clients has to be the same as attribute length. {CLIENTS} clients and {len(ATTRIBUTES)} attributes found.')


#load data
data = []
targets = []
for attribute in ATTRIBUTES:
    if targets == []:
        X, y = helper.get_samples("covid", NUM_OF_SAMPLES, [attribute], "", True, max_samples=MAX_SAMPLES)
        targets = tf.data.Dataset.from_tensor_slices(y).batch(BATCH_SIZE)
        
        tf_dataset = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
        data.append(tf_dataset)

    else:
        X, _ = helper.get_samples("covid", NUM_OF_SAMPLES, [attribute], "", True, max_samples=MAX_SAMPLES)
        tf_dataset = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
        data.append(tf_dataset)

#add targets as last entry
data.append(targets)
tf_dataset = tf.data.Dataset.zip(tuple(data))


match MODEL_TYPE:
    case "MLP":
        clients = []
        for _ in range(CLIENTS):
            clients.append(MLPClientModel(NUM_OF_SAMPLES, CLIENT_OUTPUT_NEURONS, (NUM_OF_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS_NEURONS, 'relu')))
        server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)

    case "LSTM":
        clients = []
        for _ in range(CLIENTS):
            clients.append(LSTMCLientModel(NUM_OF_SAMPLES, CLIENT_OUTPUT_NEURONS, (NUM_OF_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS_NEURONS)))
        server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)
    case "CNN":
        clients = []
        for _ in range(CLIENTS):
            clients.append(CNNClientModel(NUM_OF_SAMPLES, CLIENT_OUTPUT_NEURONS,(NUM_OF_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS_NEURONS, 3, 'relu')))
        server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)
    case _:
        raise Exception(f'Unkown model type "{MODEL_TYPE}"')


#initialize models
clients = []
for i in range(CLIENTS):
    clients.append(LSTMCLientModel(10, CLIENT_OUTPUT_NEURONS, (2, 32)))
server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)


#build model
outputs = []
for i in range(CLIENTS):
    input_data = list(tf_dataset.take(1).as_numpy_iterator())[0][i] #type:ignore
    outputs.append(clients[i](input_data))
server_input = tf.concat(outputs, axis=-1)
server_model(server_input)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

#add all variables to a list
all_trainable_variables = []
for model in clients:
    all_trainable_variables += model.trainable_variables
all_trainable_variables += server_model.trainable_variables

optimizer.build(all_trainable_variables)

loss_fn = tf.keras.losses.MeanAbsoluteError()

#results are saved in here
logs = []


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
        for model in clients:
            end_idx = start_idx + len(model.trainable_variables)
            optimizer.apply_gradients(zip(gradients[start_idx:end_idx], model.trainable_variables)) #type:ignore
            start_idx = end_idx
        #apply gradient to server model
        optimizer.apply_gradients(zip(gradients[start_idx:len(server_model.trainable_variables)], server_model.trainable_variables)) #type:ignore
    print(f'#{epoch} loss: {loss} | gradients: {tf.reduce_sum(gradients[0])}') #type:ignore
    logs.append([epoch, loss.numpy(), tf.reduce_sum(gradients[0]).numpy()]) #type:ignore

df = pd.DataFrame(logs, columns=["epoch", "loss", "gradient"])
print(df.head())