import numpy as np

# Function for recursive multi-step forecasting using a trained model
def recursive_multistep_forecasting(X_test, model, horizon):
    """
    Perform recursive multi-step forecasting using a trained model.

    Parameters:
    - X_test: pandas DataFrame, testing features
    - model: trained machine learning model
    - horizon: int, number of time steps to forecast into the future

    Returns:
    - preds: list, predicted values for each time step in the horizon
    """
    # The example consists of the last observed values seen
    # In practice, it is the first example from the test set
    example = X_test.iloc[0].values.reshape(1, -1)

    preds = []
    for i in range(horizon):
        pred = model.predict(example)[0]
        preds.append(pred)

        # Discard the value from the first position of the feature vector
        example = example[:, 1:]

        # Add the predicted value to the last position of the feature vector
        example = np.append(example, pred)
        example = example.reshape(1, -1)

    return preds

def targeted_forecasting(X_test, model):
    """
    Perform a single-step forecast using a trained model.

    Parameters:
    - X_test: pandas DataFrame, testing features
    - model: trained machine learning model

    Returns:
    - pred: float, predicted value for the next time step
    """
    # The example consists of the last observed values seen
    # In practice, it is the first example from the test set
    example = X_test.iloc[0].values.reshape(1, -1)

    # Predict the next value
    pred = model.predict(example)[0]

    return pred