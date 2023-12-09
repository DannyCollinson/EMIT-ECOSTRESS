from datetime import datetime
import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
import torch.nn as nn
import pickle
from data_classes import AutoEncoderWrapper
import pickle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial.distance import cdist

def pickle_load(pth: str):  # -> pickled_file_contents
    return pickle.load(open(pth, 'rb'))

def plot_mean_differences(original, reconstructed, encoding_dim):
    save_dir = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/loss plots/'

    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"updated_mean_diff_dim_{encoding_dim}_{current_time}.png"

    save_path = os.path.join(save_dir, file_name)

    differences = reconstructed - original
    mean_differences = torch.mean(differences, dim=0)

    x_vals = np.arange(len(mean_differences))
    plt.plot(x_vals, mean_differences, label='Mean Differences')
    plt.xlabel('Wavelength Indices')
    plt.ylabel('Mean Differences')
    plt.title('Mean Differences between Original and Reconstructed Data')
    plt.savefig(save_path)
    plt.close()

def plot_rmse(original, reconstructed, encoding_dim):
    save_dir = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/loss plots/'

    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"updated_rmse_dim_{encoding_dim}_{current_time}.png"

    save_path = os.path.join(save_dir, file_name)

    squared_differences = (original - reconstructed)**2

    mean_squared_differences = torch.mean(squared_differences, dim=0)

    rmse = torch.sqrt(mean_squared_differences)

    # Plot RMSE values
    x_vals = np.arange(len(rmse))
    plt.plot(x_vals, rmse, label='Root Mean Squared Error (RMSE)')
    plt.xlabel('Wavelength Indices')
    plt.ylabel('RMSE')
    plt.title('RMSE between Original and Reconstructed Data')
    plt.savefig(save_path)
    plt.close()

def evaluate_AE_reconstructions(ref, ref_recon, n, plot=False, text=False):
    print(f'\nN = {n}')
    
    rmse_244 = np.mean(((ref - ref_recon)**2)**(1/2), axis=0)
    rmse = np.mean(rmse_244)

    ref_flat = ref.reshape(-1, ref.shape[-1])
    ref_recon_flat = ref_recon.reshape(-1, ref_recon.shape[-1])

    r2_244 = r2_score(ref_flat, ref_recon_flat, multioutput="raw_values")
    r2 = np.mean(r2_244)
    
    return rmse, r2

def main():
    """
    Sandbox for testing code snippets

    ECOSTRESS (np.array): (n_pixels) temperature in Kelvins
    EMIT (np.ndarray): (n_pixels, n_spectra_channel)

    """
    # Initialize RNG
    rng = np.random.default_rng(seed=42)
    project_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/for_dim_reduction.pkl'

    # center data before running encoder
    emit_data_uncentered = torch.tensor(pickle_load(project_path), dtype=torch.float32)
    emit_data = emit_data_uncentered - emit_data_uncentered.mean(axis=0)
    print(f"The shape of the EMIT data is {emit_data.shape}")
    print("Loading data done.")

    dimensions = [1, 2, 3, 4, 5, 8, 16, 24, 32]
    vals = np.empty((len(dimensions)*5, 2))
    folds = []

    for j, en_dim in enumerate(dimensions):
        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        input_dim = 244 # size of the input dimensions
        encoding_dim = en_dim # latent embedding dimensions

        kf.get_n_splits(emit_data)
        for i, (train_index, test_index) in enumerate(kf.split(emit_data)):

            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

            train_data = emit_data[train_index]
            val_data = emit_data[test_index]

            model = AutoEncoderWrapper(input_dim, encoding_dim)
            model.fit(train_data, val_data)

            folds.append(i)

            vals[j, :] = evaluate_AE_reconstructions(
                    np.array(model.model.x), np.array(model.model.x_recon), en_dim, plot=False, text=False
                )
            
            model.model.plotLosses(en_dim)

        print("Working on new model...")

        model = AutoEncoderWrapper(input_dim, encoding_dim)

        if os.path.exists(f"/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/updated_model_weights_dim_{encoding_dim}.pth"):
            model.model.load_state_dict(
                torch.load(f"/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/updated_model_weights_dim_{encoding_dim}.pth")
                )
        else:
            model.fit(emit_data)
            torch.save(model.model.state_dict(),
                    f"/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/updated_model_weights_dim_{encoding_dim}.pth")

        model.model.eval()

        print("Model on eval mode...")
        # latent embeddings
        with torch.no_grad():
            latent_state = model.model.encoder(emit_data)
            decoded = model.model.decoder(latent_state)

        print("Plotting mean  differences...")
        plot_mean_differences(emit_data, decoded, encoding_dim)

        print("Plotting rmse...")
        plot_rmse(decoded, emit_data, encoding_dim)

        print(latent_state.shape)

        print("Outputting to files...")

        with open(f'updated_latent_embeddings_dim_{encoding_dim}.pkl', 'wb') as file:
            pickle.dump(latent_state, file)

        with open(f'updated_recon_vals_dim_{encoding_dim}.pkl', 'wb') as file:
            pickle.dump(decoded, file)

    print("Outputting to dataframe...")

    new_dims = np.repeat(dimensions, [5, 5, 5, 5, 5, 5, 5, 5, 5])

    df = pd.DataFrame(
    {
        'N': new_dims,
        'Fold': folds,
        'RMSE': vals[:, 0],
        'R^2': vals[:, 1],
    }
    )

    print(df)

    df.to_csv('metrics_ae.csv', index=False)

    print("Debug stop")

if __name__ == '__main__':
    main()