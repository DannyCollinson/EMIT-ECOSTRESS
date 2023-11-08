import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
import torch.nn as nn
import pickle
from modules.data import AutoEncoderWrapper


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def sandbox():
    """
    Sandbox for testing code snippets

    ECOSTRESS (np.array): (n_pixels) temperature in Kelvins
    EMIT (np.ndarray): (n_pixels, n_spectra_channel)

    """
    # Initialize RNG
    rng = np.random.default_rng()

    # Check if running with GPU runtime
    torch.cuda.is_available()

    # emit_002 = ncDataset('/content/drive/Shareddrives/emit-ecostress/Data/01_Finding_Concurrent_Data_UrbanHeat/EMIT_L2A_RFL_001_20230728T214106_2320914_002.nc')
    # (3891824) values np.array Temperature in Kelvins
    ecostress_data = pickle.load(open("data/ecostress_clean.pkl", "rb"))
    # (3891824, 244) (n_pixels, n_spectra_channel)
    emit_data = pickle.load(open("data/emit_clean.pkl", "rb"))

    train_data_subset, val_data_subset = torch.utils.data.random_split(emit_data, [0.8, 0.2])
    train_data = emit_data[train_data_subset.indices]
    val_data = emit_data[val_data_subset.indices]

    # Create AutoEncoder

    input_dim = 244
    encoding_dim = 16
    model = AutoEncoderWrapper(input_dim, encoding_dim)
    model.fit(train_data, val_data)

    device = torch.device("cuda:0")
    model.to(device)
    print("Debug stop")



if __name__ == '__main__':
    sandbox()