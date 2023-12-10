import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import pickle


def pickle_load(pth: str):  # -> pickled_file_contents
    """
    Load pickled file contents.

    Parameters:
    - pth (str): Path to the pickled file.

    Returns:
    - any: Contents of the pickled file.
    """
    with open(pth, 'rb') as file:
        return pickle.load(file)

def main():
    """
    Sandbox for running the Random Forest Model.

    Parameters:
    - None

    Loaded inside:
    - ECOSTRESS (np.array): (n_pixels) temperature in Kelvins
    - EMIT (np.ndarray): (n_pixels, n_spectra_channel)

    Returns:
    - None; will print RMSE and MAE values.

    """

    # Please add your own path to access your data files
    # Spectra data path
    emit_train_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/reflectance_train.pkl'
    emit_val_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/reflectance_val.pkl'
    emit_test_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/reflectance_test.pkl'

    # Temperature data path
    eco_train_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/temp_train.pkl'
    eco_val_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/temp_val.pkl'
    eco_test_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/temp_test.pkl'

    # Loading data
    emit_data_train = torch.tensor(pickle_load(emit_train_path), dtype=torch.float32)
    emit_data_val = torch.tensor(pickle_load(emit_val_path), dtype=torch.float32)
    emit_data_test = torch.tensor(pickle_load(emit_test_path), dtype=torch.float32)

    eco_data_train = torch.tensor(pickle_load(eco_train_path), dtype=torch.float32)
    eco_data_val = torch.tensor(pickle_load(eco_val_path), dtype=torch.float32)
    eco_data_test = torch.tensor(pickle_load(eco_test_path), dtype=torch.float32)

    # Reshaping 3D tensors into 2D so it can be input into model
    X = emit_data_train.reshape((emit_data_train.shape[0] * emit_data_train.shape[1], emit_data_train.shape[2]))
    X_val = emit_data_val.reshape((emit_data_val.shape[0] * emit_data_val.shape[1], emit_data_val.shape[2]))

    # Reshaping 2D tensors into 1D so it can be input into model
    y = eco_data_train.reshape((eco_data_train.shape[0] * eco_data_train.shape[1]))
    y_val = eco_data_val.reshape((eco_data_val.shape[0] * eco_data_val.shape[1]))

    print("Loading data done.")

    print("Creating model...")
    model = LinearRegression()

    print("Fitting...")
    model.fit(X.numpy(), y.numpy())

    print("Predicting...")
    eco_predictions_train = model.predict(X.numpy())
    eco_predictions_val = model.predict(X_val.numpy())

    # Evaluate the model using mean absolute error
    mae_train = mean_absolute_error(y, eco_predictions_train)
    mae_val = mean_absolute_error(y_val, eco_predictions_val)

    # Evaluate the model using mean squared error
    mse_train = mean_squared_error(y, eco_predictions_train)
    mse_val = mean_squared_error(y_val, eco_predictions_val)

    # Evaluate the model using root squared mean error
    rmse_train = mae_train ** 0.5
    rmse_val = mae_val ** 0.5

    # Print all the metrics
    print(f"Mean Absolute Error (Train): {mae_train}")
    print(f"Mean Absolute Error (Validation): {mae_val}")
    print(f"Mean Squared Error (Train): {mse_train}")
    print(f"Mean Squared Error (Validation): {mse_val}")
    print(f"RMSE (Train): {rmse_train}")
    print(f"RMSE (Validation): {rmse_val}")

if __name__ == '__main__':
    main()
