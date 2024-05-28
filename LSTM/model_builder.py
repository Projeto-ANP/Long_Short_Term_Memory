import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dropout, Dense
from kerastuner import HyperModel
import gc
import keras.backend as K 

class ModelBuilder(HyperModel):
    def build(self, hp):
        
        gc.collect() 
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        params = {
            'window': 12,
            'val_dropout': hp.Choice('val_dropout', values=[0.02, 0.04, 0.08]),
            'num1_lstm': hp.Choice('num1_lstm', values=[6, 12, 18, 24]),
            'num2_lstm': hp.Choice('num2_lstm', values=[6, 12, 18, 24]),
            'optimizer': hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam']),
            'activation': hp.Choice('activation', values=['linear', 'relu', 'sigmoid', 'tanh', 'selu', 'elu', 'mish']),
            'activation_dense': hp.Choice('activation_dense', values=['linear', 'relu', 'sigmoid', 'tanh', 'selu', 'elu'])
        }
        
        strategy = tf.distribute.get_strategy() 
        with strategy.scope():
            model = self.create_model(params)
        
        return model

    def create_model(self, params):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        model = Sequential([
            Input(shape=(params['window'], 1)), 
            LSTM(params['num1_lstm'], activation=params['activation'], return_sequences=True),
            Dropout(params['val_dropout']),
            LSTM(params['num2_lstm'], activation=params['activation']),
            Dropout(params['val_dropout']),
            Dense(1, activation=params['activation_dense'])
        ])

        model.compile(optimizer=params['optimizer'], loss='mse', metrics=[
            tf.keras.metrics.MeanAbsoluteError(), 
            tf.keras.metrics.RootMeanSquaredError(), 
            tf.keras.metrics.MeanSquaredError()
        ])
        
        return model