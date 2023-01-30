import tensorflow as tf

#MLP
class MLPClientModel(tf.keras.Model):
    def __init__(self, inputs: int, output_neurons: int,  hidden_layer_parameters = (2, 32,'relu')):
        super().__init__()

        self.input_layer = tf.keras.layers.Dense(32, input_shape=(inputs,), activation='relu')
        self.hidden_layers = []
        #build hidden layers
        for i in range(hidden_layer_parameters[0]):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_layer_parameters[1], activation=hidden_layer_parameters[2]))

        self.output_layer = tf.keras.layers.Dense(output_neurons, activation='linear')

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x


class MLPServerModel(tf.keras.Model):
    def __init__(self, inputs:int):
        super().__init__()

        self.input_layer = tf.keras.layers.Dense(32, input_shape=(inputs,), activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)

        return x


#LSTM
class LSTMCLientModel(tf.keras.Model):
    def __init__(self, inputs: int, output_neurons: int, hidden_layer_parameters = (2,32)):
        super().__init__()

        self.input_layer = tf.keras.layers.Reshape((inputs, 1), input_shape=(inputs,))
        self.hidden_layers = []
        for _ in range(hidden_layer_parameters[0] - 1):
            layer = tf.keras.layers.LSTM(hidden_layer_parameters[1], return_sequences=True)
            self.hidden_layers.append(layer)
        self.hidden_layers.append(tf.keras.layers.LSTM(hidden_layer_parameters[1]))
        self.output_layer = tf.keras.layers.Dense(output_neurons, activation='relu')

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x
