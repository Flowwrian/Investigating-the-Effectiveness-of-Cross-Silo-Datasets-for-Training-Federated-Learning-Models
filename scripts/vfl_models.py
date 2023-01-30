import tensorflow as tf

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