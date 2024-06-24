# import the necessary packages
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


class MachineLearning:
    def __init__(self, input_dim: int, output_dim: int = 1, num_hidden: int = 1, hidden_dim: int = 10, \
        activation: str = "relu", lr: float = 0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.lr = lr
        self.model = self.create_model()
        self.history = None

    # Create ANN using Keras
    def create_model(self) -> Sequential:
        ### Create a simple ANN model using Keras ###
        model = Sequential()
        for _ in range(self.num_hidden):
            model.add(Dense(self.hidden_dim, activation=self.activation))
        model.add(Dense(self.output_dim))
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="mean_squared_error")
        return model

    # Train the model
    def train(self, x_train, y_train, epochs: int = 100, batch_size: int = 32, \
              validation_split: float = 0.33, verbose: int = 2):
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, \
                                      validation_split=validation_split, verbose=verbose)
    
    # Loss curve plot
    def loss_curve(self):
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.show()

    def save_model(self, filename: str):
        ### Save the model ###
        self.model.save(filename)


