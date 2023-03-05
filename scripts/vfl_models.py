import tensorflow as tf
import math

#MLP
class MLPClientModel(tf.keras.Model):
    """
    A Multi-layer percetron client model.\n
    Args:
        `inputs` (int): number of inputs for the model
        `output_neurons` (int): number of output neurons of the model
        `hidden_layer_parameters` (tuple): A tuple containing specific parameters for the hidden layers
            `hidden_layer_parameters[0]` (int): number of hidden layers
            `hidden_layer_parameters[1]` (int): number of neurons per layer
            `hidden_layer_parameters[2]` (str): activation function
    """
    def __init__(self, inputs: int, output_neurons: int,  hidden_layer_parameters = (2,'relu')):
        super().__init__()
        self.optimal_neurons = [184, 344]
        self.input_layer = tf.keras.layers.Dense(416, input_shape=(inputs,), activation='relu')
        self.hidden_layers = []
        #build hidden layers
        for i in range(hidden_layer_parameters[0]):
            try:
                self.hidden_layers.append(tf.keras.layers.Dense(self.optimal_neurons[i], activation=hidden_layer_parameters[1]))
            except:
                self.hidden_layers.append(tf.keras.layers.Dense(32, activation=hidden_layer_parameters[1]))

        self.output_layer = tf.keras.layers.Dense(output_neurons, activation='linear')

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x


class MLPServerModel(tf.keras.Model):
    """
    A Multi-layer percetron server model. Contains a dense input layer with 32 neurons and a dense output layer with 1 neuron.\n
    Args:
        `inputs` (int): number of inputs for the model; Must be equal to the sum of the output neurons of the clients
    """
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
    """
    A Long Short-Term Memory client model.\n
    Args:
        `inputs` (int): number of inputs for the model
        `output_neurons` (int): number of output neurons of the model
        `hidden_layer_parameters` (tuple): A tuple containing specific parameters for the hidden layers
            `hidden_layer_parameters[0]` (int): number of hidden layers
            `hidden_layer_parameters[1]` (int): number of neurons per layer
    """
    def __init__(self, inputs: int, output_neurons: int, num_of_hidden_layers = 2):
        super().__init__()

        self.input_layer = tf.keras.layers.Reshape((10, math.floor(inputs/10)), input_shape=(inputs,))
        self.hidden_layers = []
        for _ in range(num_of_hidden_layers - 1):
            layer = tf.keras.layers.LSTM(64, return_sequences=True)
            self.hidden_layers.append(layer)
        self.hidden_layers.append(tf.keras.layers.LSTM(96))
        self.output_layer = tf.keras.layers.Dense(output_neurons, activation='relu')

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x



#CNN

class CNNClientModel(tf.keras.Model):
    """
    A Convolutional Neural-Network client model.\n
    Args:
        `inputs` (int): number of inputs for the model
        `output_neurons` (int): number of output neurons of the model
        `hidden_layer_parameters` (tuple): A tuple containing specific parameters for the hidden layers
            `hidden_layer_parameters[0]` (int): number of hidden layers
            `hidden_layer_parameters[1]` (int): number of filters per layer
            `hidden_layer_parameters[2]` (int): kernel size
            `hidden_layer_parameters[3]` (str): activation function
    """
    def __init__(self, inputs: int, output_neurons: int, hidden_layer_parameters = (2,'relu')):
        super().__init__()
        self.input_layer = tf.keras.layers.Reshape((10, math.floor(inputs/10)), input_shape=(inputs,))
        self.hidden_layers = []
        self.filters = [64, 128, 32]
        self.kernel_size = [2, 4, 4]

        for i in range(hidden_layer_parameters[0]):
            try:
                layer = tf.keras.layers.Conv1D(self.filters[i], self.kernel_size[i], padding="same", activation=hidden_layer_parameters[1])
                self.hidden_layers.append(layer)
            except:
                layer = tf.keras.layers.Conv1D(32, 3, padding="same", activation=hidden_layer_parameters[1])
                self.hidden_layers.append(layer)
        self.flatten = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(output_neurons, activation='relu')

    def call(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.output_layer(x)

        return x