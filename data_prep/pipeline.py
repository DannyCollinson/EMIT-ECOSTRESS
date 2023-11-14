import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def import_model(model_name):
    module = __import__(model_name)
    return getattr(module, 'create_model')

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32, optimizer='adam', loss='mse'):

    model.compile(optimizer=optimizer, loss=loss)

    training = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

    # loss plots but need a way to call it
    plt.plot([], label='Training Loss')
    plt.plot([], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

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


# enter parameters
use_preprocessed_data = True

path = ""
batch_amt = 10
epoch_size = 10
optimizer_type = 'adam'
loss_type = 'mse'

if use_preprocessed_data:
    preprocessed_data_file = 'path/to/preprocessed_data.pkl'
    data = load_preprocessed_data(preprocessed_data_file)
else:
    data = load_and_merge_pickle(path)

x_train, x_val, y_train, y_val = [] # what function to split(data )

with open('model_list.txt', 'r') as file:
    model_names = file.read().splitlines()

for model_name in model_names:
    create_model = import_model(model_name)
    model = create_model(input_shape=x_train.shape[1:])
    trained_model = train_model(model, x_train, y_train, x_val, y_val, epochs=epoch_size, batch_size=batch_amt, optimizer=optimizer_type, loss=loss_type)

