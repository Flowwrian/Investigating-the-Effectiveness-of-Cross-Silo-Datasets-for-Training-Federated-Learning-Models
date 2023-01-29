
import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_regression

from models import MLPClientModel, MLPServerModel

CLIENTS = 2
CLIENT_OUTPUT_NEURONS = 16

#create dataset
X, y = make_regression(1000, 10) #type:ignore
data1 = X[:500]
data2 = X[500:]
targets = y
data = [data1, data2]

tf_dataset1 = tf.data.Dataset.from_tensor_slices(data1).batch(32)
tf_dataset2 = tf.data.Dataset.from_tensor_slices(data2).batch(32)


#initialize models
clients = []
for i in range(CLIENTS):
    clients.append(MLPClientModel(10, CLIENT_OUTPUT_NEURONS, 2))
server_model = MLPServerModel(CLIENT_OUTPUT_NEURONS*CLIENTS)


#build model
outputs = []
for i in range(CLIENTS):
    outputs.append(clients[i](data[i]))
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
for epoch in range(100):
    #forward
    with tf.GradientTape() as tape:
        outputs = []
        #client output
        for i, model in enumerate(clients):
            outputs.append(model(data[i]))

        server_input = tf.concat(outputs, axis=-1)
        server_output = server_model(server_input)
        loss = loss_fn(targets, server_output)

    #compute gradients
    gradients = tape.gradient(loss, all_trainable_variables)
    print(f'#{epoch} loss: {loss} | gradients: {tf.reduce_sum(gradients[0])}') #type:ignore
    logs.append([epoch, loss.numpy(), tf.reduce_sum(gradients[0]).numpy()]) #type:ignore

    #apply gradients
    start_idx = 0
    for model in clients:
        end_idx = start_idx + len(model.trainable_variables)
        optimizer.apply_gradients(zip(gradients[start_idx:end_idx], model.trainable_variables)) #type:ignore
        start_idx = end_idx
    #apply gradient to server model
    optimizer.apply_gradients(zip(gradients[start_idx:len(server_model.trainable_variables)], server_model.trainable_variables)) #type:ignore

df = pd.DataFrame(logs, columns=["epochs", "loss", "gradient"])
print(df.head())