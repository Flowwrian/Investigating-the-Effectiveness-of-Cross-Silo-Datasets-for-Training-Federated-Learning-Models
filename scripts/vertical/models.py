import tensorflow as tf

class MLPClientModel(tf.keras.Model):
    def __init__(self, inputs: int, output_neurons: int, num_of_hidden_layers: int):
        super(MLPClientModel, self).__init__()

        self.input_layer = tf.keras.layers.Dense(32, input_shape=(inputs,), activation='relu')
        self.hidden_layers = []
        #build hidden layers
        for i in range(num_of_hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(32, activation='relu'))

        self.output_layer = tf.keras.layers.Dense(output_neurons, activation='linear')

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x


class MLPServerModel(tf.keras.Model):
    def __init__(self, inputs:int):
        super(MLPServerModel, self).__init__()

        self.input_layer = tf.keras.layers.Dense(32, input_shape=(inputs,), activation='linear')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)

        return x