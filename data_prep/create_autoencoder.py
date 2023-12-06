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

def pickle_load(pth: str):  # -> pickled_file_contents
    return pickle.load(open(pth, 'rb'))

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

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    kf.get_n_splits(emit_data)
    for i, (train_index, test_index) in enumerate(kf.split(emit_data)):

        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        train_data = emit_data[train_index]
        val_data = emit_data[test_index]

        input_dim = 244  # size of the input dimensions
        encoding_dim = 2  # latent embedding dimensions
        model = AutoEncoderWrapper(input_dim, encoding_dim)
        model.fit(train_data, val_data)
        # TODO: save loss plots to file in the class code itself
        model.model.plotLosses()

    # train_data_subset, val_data_subset = torch.utils.data.random_split(emit_data, [0.8, 0.2])
    # train_data = emit_data[train_data_subset.indices]
    # val_data = emit_data[val_data_subset.indices]

    # print(f"The shape of the EMIT train data is {train_data.shape}")
    # print(f"The shape of the EMIT val data is {val_data.shape}")
    # print(f"The data type of val data is: {val_data.dtype}")

    # Create AutoEncoder
    input_dim = 244 # size of the input dimensions
    encoding_dim = 25 # latent embedding dimensions
    model = AutoEncoderWrapper(input_dim, encoding_dim)

    if os.path.exists("/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/my_model_weights.pth"):
        model.model.load_state_dict(
            torch.load("/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/my_model_weights.pth")
            )
    else:
        model.fit(emit_data)
        torch.save(model.model.state_dict(),
                "/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/my_model_weights.pth")

    model.model.eval()

    # latent embeddings
    with torch.no_grad():
        latent_state = model.model.encoder(emit_data)
        decoded = model.model.decoder(latent_state)

    differences = decoded - emit_data
    # assert()
    mean_differences = torch.mean(differences, dim=0)

    x_vals = np.arange(len(mean_differences))
    # x_vals = torch.Tensor(np.loadtxt('/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data/EMIT_Wavelengths_20220817.txt', usecols=(1,)))
    plt.plot(x_vals, mean_differences, label='Mean Differences')
    plt.xlabel('Wavelengths')
    plt.ylabel('Mean Absolute Differences')
    plt.title('Differences in Spectrum between Original and Reconstructed Data')
    plt.show()

    print("Calculating average R2 values across samples...")
    # print("r2: " + str(model.model.generate_r2()))

    print("Plotting heatmaps of input and reconstructed input...")
    # model.model.plot_x_xrecon()

    # print(latent_state.shape)

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
    main()