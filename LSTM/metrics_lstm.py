import numpy as np
from sklearn.metrics import mean_squared_error as mse

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - float, RMSE value
    """
    return np.sqrt(mse(y_true, y_pred))

def pbe(y_true, y_pred):
    """
    Calculate Percentage Bias Error (PBE).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - float, PBE value
    """
    return 100 * (np.sum(y_true - y_pred) / np.sum(y_true))

def pocid(y_true, y_pred):
    """
    Calculate Percentage of Correctly Predicted Direction (POCID).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - float, POCID value
    """
    n = len(y_true)
    D = [1 if (y_pred[i] - y_pred[i-1]) * (y_true[i] - y_true[i-1]) > 0 else 0 for i in range(1, n)]
    POCID = 100 * np.sum(D) / n
    return POCID

def mase(y_true, y_pred, y_baseline):
    """
    Calculate Mean Absolute Scaled Error (MASE).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - y_baseline: array-like, baseline (naive) values

    Returns:
    - float, MASE value
    """
    mae_pred = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_baseline))
    result = mae_pred / mae_naive
    return result
