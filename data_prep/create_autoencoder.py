import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
import torch.nn as nn
import pickle
from data_classes import AutoEncoderWrapper
import pickle
import os

class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def load_and_merge_pickle(path):
    def join_path(relative_path: str) -> str:
        return os.path.join(path, relative_path)

    def pickle_load(relative_path: str):  # -> pickled_file_contents
        return pickle.load(open(join_path(relative_path), 'rb'))

    full_dataset = []
    lens = []
    for split_num in range(20):
        num = "0"*(2-len(str(split_num))) + str(split_num)
        full_dataset.append(
            pickle_load(
                f'reflectance_{num}_list.pkl'
            )
        )
        lens.append(full_dataset[-1].shape[0])
    full_dataset = np.concatenate(full_dataset, axis=0)

    return full_dataset

# def join_path(relative_path: str) -> str:
#     return os.path.join(base_data_path, relative_path)

def pickle_load(pth: str):  # -> pickled_file_contents
    return pickle.load(open(pth, 'rb'))

def sandbox():
    """
    Sandbox for testing code snippets

    ECOSTRESS (np.array): (n_pixels) temperature in Kelvins
    EMIT (np.ndarray): (n_pixels, n_spectra_channel)

    """
    # Initialize RNG
    rng = np.random.default_rng(seed=42)

    # Check if running with GPU runtime
    # torch.cuda.is_available()

    project_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/for_dim_reduction.pkl'

    emit_data_uncentered = torch.tensor(pickle_load(project_path), dtype=torch.float32)
    emit_data = emit_data_uncentered - emit_data_uncentered.mean(axis=0)
    print(f"The shape of the EMIT data is {emit_data.shape}")
    print("Loading data done.")
    # def pickle_save(obj: object, relative_path: str) -> None:
    #     pickle.dump(obj, open(join_path(relative_path), 'wb'))


    # emit_002 = ncDataset('/content/drive/Shareddrives/emit-ecostress/Data/01_Finding_Concurrent_Data_UrbanHeat/EMIT_L2A_RFL_001_20230728T214106_2320914_002.nc')
    # (3891824) values np.array Temperature in Kelvins
    # ecostress_data = pickle.load(open("data/ecostress_clean.pkl", "rb"))
    # (3891824, 244) (n_pixels, n_spectra_channel)
    # print("loading data...")
    # emit_data = pickle.load(open("data/emit_clean.pkl", "rb"))
    # print("done")

    # print(emit_data.shape)

    train_data_subset, val_data_subset = torch.utils.data.random_split(emit_data, [0.8, 0.2])
    train_data = emit_data[train_data_subset.indices]
    val_data = emit_data[val_data_subset.indices]

    print(f"The shape of the EMIT train data is {train_data.shape}")
    print(f"The shape of the EMIT val data is {val_data.shape}")
    print(f"The data type of val data is: {val_data.dtype}")

    # Create AutoEncoder
    input_dim = 244 # size of the input dimensions
    encoding_dim = 20 # latent embedding dimensions
    model = AutoEncoderWrapper(input_dim, encoding_dim)
    model.fit(train_data, val_data)
    model.model.plotLosses()

    print("Passing data through encoder to get latent embeddings...")
    encoder = model.model.encoder
    encoder.eval()
    latent_state = model.model.encoder(emit_data)

    print("Calculating average R2 values across samples...")
    # print("r2: " + str(model.model.generate_r2()))

    print("Plotting heatmaps of input and reconstructed input...")
    # model.model.plot_x_xrecon()

    print(latent_state.shape)

    # input_dim = 244
    # encoding_dim = 16
    # r2_array = []

    # for dim in range(1, 17):
    #     model = AutoEncoderWrapper(input_dim, dim)
    #     model.fit(train_data, val_data)
    #     r2 = model.model.generate_r2()
    #     r2_array.append(r2)
    #     print("dim: " + str(dim) + ", r2 score: " + str(r2))

    # plt.plot(range(1,17), r2_array, marker='o')
    # plt.xlabel('Encoding dimension')
    # plt.ylabel('R^2 Score')
    # plt.title('Comparing R^2 Score of Differing Encoding Dimensions')
    # plt.show()

    # device = torch.device("cuda:0")


    with open(f'latent_embeddings_dim_{encoding_dim}.pkl', 'wb') as file:
        pickle.dump(latent_state, file)

    print("Debug stop")




    # pickle file loading

    # #split for saving
    # saved = 0
    # path_template = 'Data\\Split_Data_2\\Full_PCA\\reflectance_(**)_pca244.pkl'
    # for split_num, length in enumerate(lens):
    #     pickle_save(
    #         ref_pca[saved:saved + length, :],
    #         path_template.replace(
    #             '(**)', "0"*(2-len(str(split_num))) + str(split_num)
    #         )
    #     )
    #     saved += length

if __name__ == '__main__':
    sandbox()