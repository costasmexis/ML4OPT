import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def loss_curve(history):
    plt.figure(figsize=(10, 2))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Curve")
    plt.show()


def data_scaling(df: pd.DataFrame, inputs: list, outputs: list):
    dfin = df[inputs]
    dfout = df[outputs]

    x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
    y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()

    dfin = (dfin - dfin.mean()).divide(dfin.std())
    dfout = (dfout - dfout.mean()).divide(dfout.std())

    scaled_lb = dfin.min()[inputs].values
    scaled_ub = dfin.max()[inputs].values

    x = dfin[inputs].values
    y = dfout[outputs].values

    return x, y, x_offset, x_factor, y_offset, y_factor, scaled_lb, scaled_ub


def create_nn(
    x: np.ndarray, y: np.ndarray, file_name: str = "cost_nn.keras"
) -> Sequential:
    
    model = Sequential()
    model.add(Dense(64, activation="sigmoid", input_dim=x.shape[1]))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.0005))
    history = model.fit(
        x, y, epochs=200, batch_size=64, validation_split=0.33, verbose=2
    )
    model.save(file_name)
    loss_curve(history)
    return model



