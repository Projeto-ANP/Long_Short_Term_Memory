import tensorflow as tf
import keras
import keras.backend as K # type: ignore
import keras_tuner as kt
from keras_tuner import Objective, BayesianOptimization

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import pickle
import time

import gc
import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from model_builder import ModelBuilder
from metrics_lstm import rmse, pbe, pocid, mase
from preprocessing_data import rolling_window, train_test_split_window, recursive_multistep_forecasting, convert_date # type: ignore

print(tf.config.list_physical_devices('GPU'))

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
simplefilter(action='ignore', category=FutureWarning)

#Reproducibilty
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

def rescaled_predicted_values(horizon, data, predictions, scaler, sub_dir=None, show_plot=None):
    """
    Rescale the predicted values back to the original scale and calculate evaluation metrics.

    Parameters:
    - horizon: int, prediction horizon.
    - data: DataFrame, containing the time series data used for normalization.
    - scaler: Scaler object, scaler used for normalization.
    - predictions: array-like, containing the normalized predicted values.
    - sub_dir: str or None, subdirectory to save plots (optional).
    - show_plot: bool or None, whether to display a plot of the forecasted values (optional).

    Returns:
    - rmse_rescaled (float): Rescaled Root Mean Squared Error.
    - mape_rescaled (float): Rescaled Mean Absolute Percentage Error.
    - pbe_rescaled (float): Rescaled Prediction Bias Error.
    - pocid_rescaled (float): Rescaled Percentage of Correct Indication Direction.
    - mase_rescaled (float): Rescaled Mean Absolute Scaled Error.
    - mae_rescaled (float): Rescaled Mean Absolute Error.
    """
    
    # Inverse MinMax scaling to get predictions back to original scale
    mat_predictions = np.zeros((len(predictions), 13)) 
    for i, pred in enumerate(predictions):
        mat_predictions[i, -1] = pred
    predictions_rescaled = scaler.inverse_transform(mat_predictions)[:, 12]

    # Retrieve the actual values in the original scale
    y_test_rescaled = data["m3"][-horizon:].values
    
    # Calculation of evaluation metrics
    rmse_result_rescaled = np.sqrt(mse(y_test_rescaled, predictions_rescaled))
    mape_result_rescaled = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100
    pbe_result_rescaled = 100 * np.sum((y_test_rescaled - predictions_rescaled)) / np.sum(y_test_rescaled)
    pocid_result_rescaled = 100 * np.sum((predictions_rescaled[1:] - predictions_rescaled[:-1]) * (y_test_rescaled[1:] - y_test_rescaled[:-1]) > 0) / (len(y_test_rescaled) - 1)
    y_baseline = data["m3"][-horizon*2:-horizon].values
    mase_result_rescaled = np.mean(np.abs(y_test_rescaled - predictions_rescaled)) / np.mean(np.abs(y_test_rescaled - y_baseline))
    mae_rescaled = np.mean(np.abs(y_test_rescaled - predictions_rescaled))

    # Optionally, plot the rescaled predictions
    if show_plot:
        plt.figure(figsize=(12, 3))
        plt.title('Predictions in original scale')
        plt.plot(y_test_rescaled, label='Actual')
        plt.plot(predictions_rescaled, linewidth=5, alpha=0.4, label='Predicted')
        plt.plot(y_baseline, label='Baseline')
        plt.scatter(range(len(y_test_rescaled)), y_test_rescaled, color='blue')
        plt.scatter(range(len(predictions_rescaled)), predictions_rescaled, color='red')
        plt.scatter(range(len(y_baseline)), y_baseline, color='green')
        plt.legend()
        
        # Save the plot if sub_dir is provided
        if sub_dir:
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            plt.savefig(os.path.join(sub_dir, 'rescaled_predictions.png'))
        plt.close()

    return rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled, mae_rescaled

def create_lstm_model(horizon, window, data, epochs, state, product, rescaling=None, show_plot=None, verbose=2, return_model=None):
    """
    Run LSTM model for time series forecasting.

    Parameters:
    - horizon (int): Prediction horizon.
    - window (int): Length of the window for attribute-value table generation.
    - data (pd.DataFrame): DataFrame containing the time series data.
    - epochs (int): Number of epochs for training the LSTM model.
    - state: The specific state for the model (description needed based on context).
    - product: The specific product for the model (description needed based on context).
    - rescaling (bool or None, optional): Whether to rescale the predicted values to the original scale. Default is None.
    - show_plot (bool or None, optional): Whether to display a plot of the forecasted values. Default is None.
    - verbose (int, optional): Controls the verbosity of the training process. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 2.
    - return_model (bool or None, optional): Whether to return the trained model. Default is None.
        
    Returns:
    - rmse (float): Root Mean Squared Error.
    - mape (float): Mean Absolute Percentage Error.
    - pbe (float): Prediction Bias Error.
    - pocid (float): Percentage of Correct Indication Direction.
        
    If rescaling is True, also returns rescaled metrics:
    - rmse_rescaled (float): Rescaled Root Mean Squared Error.
    - mape_rescaled (float): Rescaled Mean Absolute Percentage Error.
    - pbe_rescaled (float): Rescaled Prediction Bias Error.
    - pocid_rescaled (float): Rescaled Percentage of Correct Indication Direction.
    - mase_rescaled (float): Rescaled Mean Absolute Scaled Error.
    """
    
    # Generating the attribute-value table (normalized)
    data_normalized, scaler = rolling_window(data["m3"], window)

    # Splitting the data into train/test considering a prediction horizon of 12 months
    X_train, X_test, y_train, y_test = train_test_split_window(data_normalized, horizon)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42, shuffle=False)
    
    # Define Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', patience=10, mode='min', verbose=verbose)
    
    tuner = BayesianOptimization(
        hypermodel=ModelBuilder(window=window),
        objective=Objective('root_mean_squared_error', direction='min'),
        num_initial_points=5,
        max_trials=30,
        alpha=0.0001,
        beta=2.6,
        seed=42,
        max_retries_per_trial=1,
        max_consecutive_failed_trials=3,
        directory=f'tuner_v2_window_{window}',
        project_name=f'lstm_{state}_{product}'
    )
    
    tuner.search(X_train, y_train, epochs=epochs, 
                 validation_data=(X_val, y_val), 
                 batch_size=32,
                 verbose=verbose, callbacks=[early_stopping])
        
    # Get the best model
    best_model = tuner.get_best_models()[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0].values

    gc.collect() 
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    del tuner
    
    # Predicting
    predictions = recursive_multistep_forecasting(X_test, best_model, horizon)    

    # Calculating evaluation metrics
    rmse_result = rmse(y_test.values, predictions)
    mape_result = mape(y_test.values, predictions)
    pbe_result = pbe(y_test.values, predictions)
    pocid_result = pocid(y_test.values, predictions)
    
    sub_dir = None

    if show_plot:
        plots_dir = "plots_graphs"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        sub_dir = os.path.join(plots_dir, f"plot_{state}_{product}")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        plt.figure(figsize=(12, 3))
        plt.title('Normalized Predictions')
        plt.plot(y_test.values, label='Actual')
        plt.plot(predictions, linewidth=5, alpha=0.4, label='Predicted')
        plt.scatter(range(len(y_test)), y_test.values, color='blue')
        plt.scatter(range(len(predictions)), predictions, color='red')
        plt.legend()
        plt.savefig(os.path.join(sub_dir, 'normalized_predictions.png'))
        plt.close()

        plt.figure(figsize=(8,8))
        plt.scatter(y_test.values, predictions, alpha=0.4)
        plt.axline((0, 0), slope=1, linestyle='dotted', color='gray')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.savefig(os.path.join(sub_dir, 'actual_vs_predicted.png'))
        plt.close()
        
    if rescaling:
        if return_model:
            rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled = rescaled_predicted_values(horizon=horizon, data=data, predictions=predictions, scaler=scaler, sub_dir=sub_dir, show_plot=show_plot)
            return best_model, rmse_result, mape_result, pbe_result, pocid_result, best_hyperparameters, rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled
        else:
            rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled = rescaled_predicted_values(horizon=horizon, data=data, predictions=predictions, scaler=scaler, sub_dir=sub_dir, show_plot=show_plot)
        return rmse_result, mape_result, pbe_result, pocid_result, best_hyperparameters, rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled
    else:
        if return_model:
            return best_model, rmse_result, mape_result, pbe_result, pocid_result, best_hyperparameters
        else:
            return rmse_result, mape_result, pbe_result, pocid_result, best_hyperparameters    

def run_lstm(state, product, horizon, window, data_filtered, epochs, verbose, save_model, bool_save):
    """
    Run LSTM model training and save results to an Excel file.

    Parameters:
        - state (str): State for which LSTM model is trained.
        - product (str): Product for which LSTM model is trained.
        - horizon (int): Prediction horizon.
        - window (int): Length of the window for attribute-value table generation.
        - data_filtered (pd.DataFrame): Filtered data specific to the state and product.
        - epochs (int): Number of epochs for training the LSTM model.
        - verbose (int): Controls the verbosity of the training process. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        - save_model (bool): Flag indicating whether to save the trained model.
        - bool_save (bool): Flag indicating whether to save results to an Excel file.

    Returns:
        None
    """

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.keras.utils.set_random_seed(42)

    try:
        # Run LSTM model training
        model, rmse_result, mape_result, pbe_result, pocid_result, best_hyperparameters, rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled = \
        create_lstm_model(horizon=horizon, window=window, data=data_filtered, epochs=epochs, state=state, product=product, rescaling=True, show_plot=False, verbose=verbose, return_model=True)

        # Save trained model if specified
        if save_model:
            if model is not None:
                directory = "saved_models"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                state_directory = os.path.join(directory, state)
                if not os.path.exists(state_directory):
                    os.makedirs(state_directory)

                model_name = f"lstm_{state}_{product}_window_{window}"
                model_path = os.path.join(state_directory, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Prepare results into a DataFrame
        results_df = pd.DataFrame([{'HORIZON': horizon,
                                    'WINDOW': window,
                                    'EPOCHS': epochs,
                                    'BEST_PARAM': str(best_hyperparameters),
                                    'VAL_DROPOUT': best_hyperparameters['val_dropout'],
                                    'NUM1_LSTM': best_hyperparameters['num1_lstm'],
                                    'NUM2_LSTM': best_hyperparameters['num2_lstm'],
                                    'OPTIMIZER':"nadam",
                                    'ACTIVATION': best_hyperparameters['activation'],
                                    "ACTIVATION_DENSE": best_hyperparameters['activation_dense'],
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': rmse_result,
                                    'MAPE': mape_result,
                                    'PBE': pbe_result,
                                    'POCID': pocid_result,
                                    'RMSE_RESCALED': rmse_result_rescaled,
                                    'MAPE_RESCALED': mape_result_rescaled,
                                    'PBE_RESCALED': pbe_result_rescaled,
                                    'POCID_RESCALED': pocid_result_rescaled,
                                    'MASE_RESCALED': mase_result_rescaled,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle any exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'HORIZON': np.nan,
                                    'WINDOW': np.nan,
                                    'EPOCHS': np.nan,
                                    'BEST_PARAM': np.nan,
                                    'VAL_DROPOUT': np.nan,
                                    'NUM1_LSTM': np.nan,
                                    'NUM2_LSTM': np.nan,
                                    'OPTIMIZER': np.nan,
                                    'ACTIVATION': np.nan,
                                    "ACTIVATION_DENSE": np.nan,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'RMSE_RESCALED': np.nan,
                                    'MAPE_RESCALED': np.nan,
                                    'PBE_RESCALED': np.nan,
                                    'POCID_RESCALED': np.nan,
                                    'MASE_RESCALED': np.nan,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            
    # Save results to an Excel file if specified
    if bool_save:
        directory = f'result_v2_{window}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, 'lstm_results.xlsx')
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
        else:
            existing_df = pd.DataFrame()

        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_excel(file_path, index=False)

def run_lstm_in_thread(horizon, window, epochs, verbose, bool_save, save_model=None):
    """
    Loop through LSTM model with different configurations for each state and product combination.

    Parameters:
        - horizon (int): Prediction horizon.
        - window (int): Length of the window for attribute-value table generation.
        - epochs (int): Number of epochs for training the LSTM model.
        - verbose (int): Controls the verbosity of the training process. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        - bool_save (bool): Flag indicating whether to save the trained models.
        - save_model (bool or None, optional): Save models.

    Returns:
        None
    """
    
    multiprocessing.set_start_method("spawn")

    # Load the combined dataset
    all_data = pd.read_csv('../database/combined_data.csv', sep=";")

    # Initialize a dictionary to store products for each state
    state_product_dict = {}

    # Iterate over unique states
    for state in all_data['state'].unique():
        # Filter products corresponding to this state
        products = all_data[all_data['state'] == state]['product'].unique()
        # Add to the dictionary
        state_product_dict[state] = list(products)

    # Loop through each state and its products
    for state, products in state_product_dict.items():
        for product in products:
            print(f"========== State: {state}, product: {product} ==========")

            # Record the start time of execution
            start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nExecution started at: {start_timestamp}")
            start_time = time.time()

            # Set random seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            os.environ['PYTHONHASHSEED'] = str(42)
            tf.keras.utils.set_random_seed(42)

            # Filter data for the current state and product
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == product)]

            # Create a separate process (thread) to run LSTM model
            thread = multiprocessing.Process(target=run_lstm, args=(state, product, horizon, window, data_filtered, epochs, verbose, save_model, bool_save))
            thread.start()
            thread.join()  # Wait for the thread to finish execution
    
            # Calculate and print the execution time
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Function execution time: {execution_time:.2f} seconds")
            print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def product_and_single_thread_testing():    
    """
    Perform a simple training thread using LSTM model for time series forecasting.

    This function initializes random seeds, loads a database, executes an LSTM model,
    evaluates its performance, and prints results.

    Parameters:
    None

    Returns:
    None
    """
    # Setting random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.keras.utils.set_random_seed(42)
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv("../database/venda_process/mensal/uf/glp/mensal_pr_glp.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(" ========== Starting univariate test for the state of Paraná - GLP ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the LSTM model
    rmse_result, mape_result, pbe_result, pocid_result, best_param, rmse_result_rescaled, mape_result_rescaled, pbe_result_rescaled, pocid_result_rescaled, mase_result_rescaled = \
    create_lstm_model(horizon=12, window=12, data=data_filtered_test, epochs=200, state="pr", product="glp", rescaling=True, show_plot=True, verbose=1)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Printing results without scaling
    print("\nResults without scaling:")
    print(f'RMSE: {rmse_result}')
    print(f'MAPE: {mape_result}')
    print(f'PBE: {pbe_result}')
    print(f'POCID: {pocid_result}')

    # Printing rescaled results
    print("\nRescaled results:")
    print(f'Rescaled RMSE: {rmse_result_rescaled}')
    print(f'Rescaled MAPE: {mape_result_rescaled}')
    print(f'Rescaled PBE: {pbe_result_rescaled}')
    print(f'Rescaled POCID: {pocid_result_rescaled}')
    print(f'Rescaled MASE: {mase_result_rescaled}')

    # Displaying the best parameters found during model tuning
    print("\nBest parameters found: ", best_param)