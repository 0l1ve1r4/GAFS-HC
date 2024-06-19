import tensorflow as tf
from tensorflow import keras
from .utils import *
import numpy as np

class NeuralNetwork:

    def __init__(self, num_features):
        self.num_features = int(num_features)
        self.model = self._create_model()
        
    def _create_model(self):
        model = keras.Sequential()
        
        entry_neurons = self.num_features
        output_neurons = 1
        hidden_neurons = (entry_neurons + output_neurons)//2
        
        model.add(keras.layers.Dense(input_shape=(self.num_features,), units=entry_neurons, activation='linear'))
        model.add(keras.layers.Dense(units=hidden_neurons, activation='relu'))
        model.add(keras.layers.Dense(units=output_neurons, activation='linear'))
        
        model.compile(optimizer='adam', loss='mape')

        return model

    def read_file(self, filename):
        """Read a file with binary lists and fitness values."""

        data = []
        fitness_values = []
        with open(filename, "r") as f:
            for line in f:
                
                line = line.strip().replace("[", "").replace("]", "").split(",")
                binary_list = line[:-1]
                binary_list = [int(x) for x in binary_list]
                
                fitness_value = float(line[-1])
                data.append(binary_list)
                fitness_values.append(fitness_value)
        
        return data, fitness_values

    def train_nn(self, training_data_path: str, epochs=100):

        individuals, fitness = self.read_file(training_data_path)
        X_train = np.array(individuals)
        Y_train = np.array(fitness)
        X_train = X_train.reshape(-1, self.num_features)
        Y_train = Y_train.reshape(-1, 1)

        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=1)
    
        
    def save_nn(self, file_path: str):
        """Save the trained neural network model to a file."""

        self.model.save(file_path)

    def load_nn(self, file_path: str):
        """Load a pre-trained neural network model from a file."""

        self.model = keras.models.load_model(file_path)

    def evaluate_fitness(self, binary_list):
        """Evaluate the fitness of a binary list using the trained model."""

        data = tf.cast(binary_list, dtype=tf.float32)
        data = tf.expand_dims(data, axis=0)
        fitness = self.model.predict(data, verbose=0)[0][0]
        return fitness
    
    def evaluate_population(self, population):
        """Evaluate the fitness of each binary list in a population."""
    
        fitness = (self.model.predict(np.array(population), verbose=0)).tolist()
                
        return [x[0] for x in fitness]
        
    def evaluate_list_of_lists(self, list_of_binary_lists):
        """Evaluate the fitness of each binary list in a list of lists."""
        
        fitness_values = []
        for binary_list in list_of_binary_lists:
            fitness = self.evaluate_fitness(binary_list)
            fitness_values.append(fitness)
        return fitness_values