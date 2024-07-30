import tensorflow as tf
from keras import Sequential, Input
from keras.layers import LSTM, Dropout, Dense # type: ignore
from kerastuner import HyperModel
import gc
import keras.backend as K  # type: ignore

class ModelBuilder(HyperModel):
    def __init__(self, window, len_predictions):
        """
        Initializes the ModelBuilder object.

        Args:
        - window (int): Size of the input window for the model.
        - len_predictions (int): number of predictions to be made.
        """
        self.window = window
        self.len_predictions = len_predictions

    def build(self, hp):
        """
        Builds a deep learning model based on hyperparameter choices.

        Args:
        - hp: Hyperparameters object from Kerastuner.

        Returns:
        - model: Compiled Keras model.
        """
        # Garbage collection, session clearing, and graph reset for memory management
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()

        # Define hyperparameters
        params = {
            'window': self.window,
            'val_dropout': hp.Choice('val_dropout', values=[0.01]),
            'num1_lstm': hp.Choice('num1_lstm', values=list(range(48, 205, 12))),
            'num2_lstm': hp.Choice('num2_lstm', values=list(range(48, 205, 12))),
            'activation': hp.Choice('activation', values=['selu']),
            'activation_dense': hp.Choice('activation_dense', values=['elu'])
        }

        # Get the distribution strategy for multi-GPU training
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            model = self.create_model(params)

        return model

    def create_model(self, params):
        """
        Creates a Keras sequential model based on given parameters.

        Args:
        - params (dict): Dictionary containing model configuration parameters.

        Returns:
        - model: Compiled Keras model.
        """
        # Set memory growth for GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Define the model architecture
        model = Sequential([
            Input(shape=(params['window'], 1)),
            LSTM(params['num1_lstm'], activation=params['activation'], return_sequences=True),
            Dropout(params['val_dropout']),
            LSTM(params['num2_lstm'], activation=params['activation']),
            Dropout(params['val_dropout']),
            Dense(self.len_predictions, activation=params['activation_dense'])
        ])

        # Compile the model with optimizer, loss, and metrics
        model.compile(optimizer='nadam', loss='mse', metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanSquaredError()
        ])

        return model