import numpy as np
import pickle
from models import RandomForestWrapper
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import pickle


def pickle_load(pth: str):  # -> pickled_file_contents
    with open(pth, 'rb') as file:
        return pickle.load(file)
    

def plot_loss(train_losses, val_losses, output_folder):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training MAE')
    plt.plot(epochs, val_losses, label='Validation MAE')
    plt.title('Training and Validation MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, 'random_forest_mae_curves.png')
    plt.savefig(output_path)
    plt.show()

def main():
    """
    Sandbox for testing code snippets

    ECOSTRESS (np.array): (n_pixels) temperature in Kelvins
    EMIT (np.ndarray): (n_pixels, n_spectra_channel)

    """
    rng = np.random.default_rng(seed=42)

    emit_train_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/reflectance_train.pkl'
    emit_val_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/reflectance_val.pkl'
    emit_test_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/reflectance_test.pkl'

    eco_train_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/temp_train.pkl'
    eco_val_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/temp_val.pkl'
    eco_test_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/temp_test.pkl'

    emit_data_train = torch.tensor(pickle_load(emit_train_path), dtype=torch.float32)
    emit_data_val = torch.tensor(pickle_load(emit_val_path), dtype=torch.float32)
    emit_data_test = torch.tensor(pickle_load(emit_test_path), dtype=torch.float32)

    eco_data_train = torch.tensor(pickle_load(eco_train_path), dtype=torch.float32)
    eco_data_val = torch.tensor(pickle_load(eco_val_path), dtype=torch.float32)
    eco_data_test = torch.tensor(pickle_load(eco_test_path), dtype=torch.float32)

    print(f"The shape of the EMIT data is {emit_data_train.shape}")
    print("Loading data done.")

    print("Loading data done.")

    input_dim = 244
    output_dim = 1

    model = RandomForestWrapper(input_dim, output_dim)

    weights_path = f"/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/rf_weights.pth"

    if os.path.exists(weights_path):
        # Load the trained model from file
        model.model = pickle_load(weights_path)
    else:
        # Train the model
        model.fit(emit_data_train.numpy(), eco_data_train.numpy())

        # Save the trained model
        model.save(weights_path)

    # Make predictions
    eco_predictions_train = model.predict(emit_data_train.numpy())
    eco_predictions_val = model.predict(emit_data_val.numpy())

    # Evaluate the model
    mae_train = mean_absolute_error(eco_data_train, eco_predictions_train)
    mae_val = mean_absolute_error(eco_data_val, eco_predictions_val)

    print(f"Mean Absolute Error (Train): {mae_train}")
    print(f"Mean Absolute Error (Validation): {mae_val}")

    plot_loss([mae_train], [mae_val], '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/loss plots/')

if __name__ == '__main__':
    main()
