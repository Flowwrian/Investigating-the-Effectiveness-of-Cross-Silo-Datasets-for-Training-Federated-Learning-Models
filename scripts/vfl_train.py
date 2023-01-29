import pandas as pd
import tensorflow as tf

from vfl_models import MLPClientModel, MLPServerModel
import helper

CLIENTS = 2
CLIENT_OUTPUT_NEURONS = 16
ATTRIBUTES = ["new_cases", "weekly_hosp_admissions"]
NUM_OF_SAMPLES = 10
BATCH_SIZE = 500
MAX_SAMPLES = 30000
EPOCHS = 10

if CLIENTS != len(ATTRIBUTES):
    Exception(f'Clients has to be the same as attribute length. {CLIENTS} clients and {len(ATTRIBUTES)} attributes found.')


#load data
data = []
targets = []
for attribute in ATTRIBUTES:
    if targets == []:
        X, y = helper.get_samples("covid", NUM_OF_SAMPLES, [attribute], "", True, max_samples=MAX_SAMPLES)
        targets = list(tf.data.Dataset.from_tensor_slices(y).batch(BATCH_SIZE).as_numpy_iterator())
        
        tf_dataset = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
        data.append(list(tf_dataset.as_numpy_iterator()))

    else:
        X, _ = helper.get_samples("covid", NUM_OF_SAMPLES, [attribute], "", True, max_samples=MAX_SAMPLES)
        tf_dataset = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
        data.append(list(tf_dataset.as_numpy_iterator()))
num_of_batches = len(targets)

#initialize models
clients = []
for i in range(CLIENTS):
    clients.append(MLPClientModel(10, CLIENT_OUTPUT_NEURONS, 2))
server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)


#build model
outputs = []
for i in range(CLIENTS):
    input_data = data[i][0]
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
    for batch in range(num_of_batches):
        with tf.GradientTape() as tape:
            outputs = []
            #client output
            for i, model in enumerate(clients):
                outputs.append(model(data[i][batch]))

            server_input = tf.concat(outputs, axis=-1)
            server_output = server_model(server_input)
            loss = loss_fn(targets[batch], server_output)

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