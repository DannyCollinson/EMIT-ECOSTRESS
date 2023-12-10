import datetime
import numpy as np
import torch
import pickle
from models import TransformerWrapper
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any, List

def pickle_load(pth: str): # -> pickled_file_contents
    """
    Load pickled file contents.

    Parameters:
    - pth (str): Path to the pickled file.

    Returns:
    - any: Contents of the pickled file.
    """
    with open(pth, 'rb') as file:
        return pickle.load(file)
    

def plot_loss(train_losses: List[float], val_losses: List[float], output_folder: str) -> None:
    """
    Plot and save the training and validation loss curves.

    Parameters:
    - train_losses (List[float]): Training losses.
    - val_losses (List[float]): Validation losses.
    - output_folder (str): Output folder for saving the plot.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, 'transformer_loss_curves.png')
    plt.savefig(output_path)
    plt.close()

def plot_differences(predictions: torch.Tensor, ground_truth: torch.Tensor) -> None:
    """
    Plot and save the differences in spectrum between original and reconstructed data.

    Parameters:
    - predictions (torch.Tensor): Predicted data.
    - ground_truth (torch.Tensor): Ground truth data.
    """
    # Add your own save path
    save_dir = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/transformer diff plots/'

    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"t_diff_plot_{current_time}.png"

    save_path = os.path.join(save_dir, file_name)

    differences = predictions - ground_truth

    mean_differences = torch.mean(differences, dim=0)

    x_vals = np.arange(len(mean_differences))

    plt.plot(x_vals, mean_differences, label='Mean Differences')
    plt.xlabel('Wavelength Indices')
    plt.ylabel('Mean Absolute Differences')
    plt.title('Differences in Spectrum between Original and Reconstructed Data')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """
    Sandbox for running the Transformer Model.

    Parameters:
    - None

    Loaded inside:
    - ECOSTRESS (np.array): (n_pixels) temperature in Kelvins
    - EMIT (np.ndarray): (n_pixels, n_spectra_channel)

    Returns:
    - None; Will save weights and states to the files you designate. 

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

    print("Loading data done.")

    # Initialize dimensions
    input_dim = 244
    output_dim = 1

    print("Creating model...")
    model = TransformerWrapper(input_dim, output_dim)

    # Add your own path
    weights_path = f"/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/transformer_weights.pth"

    # If model already exists, loads weights otherwise fits new model
    if os.path.exists(weights_path):
        print("Loading model...")
        model.model.load_state_dict(torch.load(weights_path))
    else:
        print("Fitting new model...")
        model.fit(emit_data_train, eco_data_val)
        torch.save(model.model.state_dict(), weights_path)

    print("Plotting losses...")
    # Add your own save path
    plot_loss(model.model.train_losses, model.model.val_losses, '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/loss plots/')

    print("Model in evaluation mode...")
    model.model.eval()

    with torch.no_grad():
        eco_predictions_train = model.model(emit_data_train)

    print("Plotting differences...")
    plot_differences(eco_data_train, eco_predictions_train)

    print("Model evaluation completed.")

if __name__ == '__main__':
    main()
