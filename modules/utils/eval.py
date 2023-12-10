import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Size


def plot_loss_patch_to_pixel(
        train_loss: np.ndarray,
        val_loss: np.ndarray,
        radius: int,
        n_dimensions: int,
        model_name: str,
        input_type: str,
) -> None:  # displays plot
    '''
    Plots the train and validation curves for a given model. Returns None
    
    Input
    train_loss: 1-D numpy array of training loss values for each epoch
    val_loss: 1-D numpy array of validation loss values for each epoch
    radius: the radius used for the patch-to-pixel model
    n_dimensions: the number of non-elevation dimensions in the model input
    model_name: the type of model used, e.g. "Small Dense Network"
    input_type: the type of input, e.g. "PCA"
    '''
    fig, ax = plt.subplots()
    fig.suptitle(
        f'{model_name}, radius={radius}'
    )
    l = 2 * radius + 1
    ax.set_title(
        f'Input = {l}x{l}x{n_dimensions}, '
        f'{input_type} + elevation'
    )
    ax.semilogy(
        np.arange(len(train_loss)),
        train_loss,
        label=(
            'train, '
            f'min std={min(train_loss):.4}, '
        ),
    )
    ax.semilogy(
        np.arange(len(val_loss)),
        val_loss,
        label=(
            'val, '
            f'min std={min(val_loss):.4}, '
        ),
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average RMSE Loss')
    ax.legend()
    plt.show(fig)
    
    
def plot_loss_cnn(
        train_loss: np.ndarray,
        val_loss: np.ndarray,
        x_size: int,
        y_size: int,
        n_dimensions: int,
        model_name: str,
        input_type: str,
) -> None:  # displays plot
    '''
    Plots the train and validation curves for a given model. Returns None
    
    Input
    train_loss: 1-D numpy array of training loss values for each epoch
    val_loss: 1-D numpy array of validation loss values for each epoch
    x_size: the x_size used for the CNN model
    y_size: the y_size used for the CNN model
    n_dimensions: the number of non-elevation dimensions in the model input
    model_name: the type of model used, e.g. "U-Net"
    input_type: the type of input, e.g. "PCA"
    '''
    fig, ax = plt.subplots()
    fig.suptitle(f'{model_name}')
    ax.set_title(
        f'Input = {x_size}x{y_size}x{n_dimensions}, '
        f'{input_type} + elevation'
    )
    ax.semilogy(
        np.arange(len(train_loss)),
        train_loss,
        label=(
            'train, '
            f'min std={min(train_loss):.4}, '
        ),
    )
    ax.semilogy(
        np.arange(len(val_loss)),
        val_loss,
        label=(
            'val, '
            f'min std={min(val_loss):.4}, '
        ),
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average RMSE Loss')
    ax.legend()
    plt.show(fig)


def plot_loss_on_map_patch_to_pixel(
        train_loss_array: np.ndarray,
        val_loss_array: np.ndarray,
        radius: int,
        n_dimensions: int,
) -> None:  # displays plot
    '''
    Plots the training and validation loss per pixel on the map
    to visualize the distribution
    
    Input
    train_loss_array: numpy array of average training RMSE per pixel
    val_loss_array: numpy array of average validation RMSE per pixel
    radius: the radius used for the model
    n_dimensions: the number of non-elevation dimensions of the model input
    '''
    fig, ax = plt.subplots()
    ax.set_title(
        f'Train RMSE on Map, Radius={radius}, N-dimensions={n_dimensions}'
    )
    plt.imshow(train_loss_array)
    plt.colorbar(fraction=0.05, shrink=0.6)
    ax.matshow(train_loss_array)
    plt.show(fig)
    fig, ax = plt.subplots()
    ax.set_title(
        f'Validation RMSE on Map, Radius={radius}, N-dimensions={n_dimensions}'
    )
    plt.imshow(train_loss_array)
    plt.colorbar(fraction=0.05, shrink=0.8)
    ax.matshow(val_loss_array)
    plt.show(fig)
    
    
def plot_loss_on_map_cnn(
        train_loss_array: np.ndarray,
        val_loss_array: np.ndarray,
        x_size: int,
        y_size: int,
        n_dimensions: int,
) -> None:  # displays plot
    '''
    Plots the training and validation loss per pixel on the map
    to visualize the distribution
    
    Input
    train_loss_array: numpy array of average training RMSE per pixel
    val_loss_array: numpy array of average validation RMSE per pixel
    radius: the radius used for the model
    n_dimensions: the number of non-elevation dimensions of the model input
    '''
    fig, ax = plt.subplots()
    ax.set_title(
        f'Train RMSE on Map, x_size,y_size={x_size},{y_size}, '
        f'N-dimensions={n_dimensions}'
    )
    plt.imshow(train_loss_array)
    plt.colorbar(fraction=0.05, shrink=0.6)
    ax.matshow(train_loss_array)
    plt.show(fig)
    fig, ax = plt.subplots()
    ax.set_title(
        f'Validation RMSE on Map, x,y={x_size},{y_size}, '
        f'N-dimensions={n_dimensions}'
    )
    plt.imshow(train_loss_array)
    plt.colorbar(fraction=0.05, shrink=0.8)
    ax.matshow(val_loss_array)
    plt.show(fig)


def initialize_eval_results() -> pd.DataFrame:
    '''
    Initializes the dataframe for the evaluation results
    '''
    return pd.DataFrame(
        columns=[
            'radius',
            'n_dimensions',
            
            'train_avg_std',
            'train_std_std',
            'train_std_min',
            'train_std_0.5pct',
            'train_std_2.5pct',
            'train_std_16pct',
            'train_std_25pct',
            'train_std_50pct',
            'train_std_75pct',
            'train_std_84pct',
            'train_std_97.5pct',
            'train_std_99.5pct',
            'train_std_max',
            
            'train_avg_K',
            'train_std_K',
            'train_K_min',
            'train_K_0.5pct',
            'train_K_2.5pct',
            'train_K_16pct',
            'train_K_25pct',
            'train_K_50pct',
            'train_K_75pct',
            'train_K_84pct',
            'train_K_97.5pct',
            'train_K_99.5pct',
            'train_K_max',
            
            'val_avg_std',
            'val_std_std',
            'val_std_min',
            'val_std_0.5pct',
            'val_std_2.5pct',
            'val_std_16pct',
            'val_std_25pct',
            'val_std_50pct',
            'val_std_75pct',
            'val_std_84pct',
            'val_std_97.5pct',
            'val_std_99.5pct',
            'val_std_max',
            
            'val_avg_K',
            'val_std_K',
            'val_K_min',
            'val_K_0.5pct',
            'val_K_2.5pct',
            'val_K_16pct',
            'val_K_25pct',
            'val_K_50pct',
            'val_K_75pct',
            'val_K_84pct',
            'val_K_97.5pct',
            'val_K_99.5pct',
            'val_K_max',
        ]
    )
    
    
def evaluate_model_performance(
    train_dataset_shape: Size,
    train_dataset_std: float,
    val_dataset_shape: Size,
    val_dataset_std: float,
    train_loss_list: list[np.ndarray],
    val_loss_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculates the evaluation statistics given the list of loss results
    from the last epoch of training and validation
    
    Input
    train_dataset_shape: the 2-D shape of the training dataset
    val_dataset_shape: the 2-D shape of the validation dataset
    train_loss_list: list of numpy arrays containing losses 
                   from the last training epoch
    val_loss_list: list of numpy arrays containing losses 
                   from the last validation epoch
    
    Returns
    stats: numpy array containing the loss statistics with values that
           match the columns from initialize_eval_results in order
           
    train_loss_vals: numpy array containing all training loss values
                     reshaped to the size of the dataset
    val_loss_vals: numpy array containing all validation loss values
                     reshaped to the size of the dataset
    '''
    train_loss_vals = np.vstack(
            [
                np.array(train_loss_list[i]).flatten()[:, np.newaxis]
                for i in range(len(train_loss_list))
            ]
    ).reshape((train_dataset_shape[0], train_dataset_shape[1]))
        
    val_loss_vals = np.vstack(
            [
                np.array(val_loss_list[i]).flatten()[:, np.newaxis]
                for i in range(len(val_loss_list))
            ]
    ).reshape((val_dataset_shape[0], val_dataset_shape[1]))
        
    train_loss_stats = (
        np.concatenate(
            [
                [train_loss_vals.mean()],
                [train_loss_vals.std()],
                np.percentile(
                    train_loss_vals,
                    [0,0.5, 2.5, 16, 25, 50, 75, 84, 97.5, 99.5, 100],
                )
            ]
        )
    )
    train_loss_stats = np.sqrt(train_loss_stats)
    train_stats = np.concatenate(
        [train_loss_stats, train_dataset_std * train_loss_stats]
    )
    
    val_loss_stats = (
        np.concatenate(
            [
                [val_loss_vals.mean()],
                [val_loss_vals.std()],
                np.percentile(
                    val_loss_vals,
                    [0,0.5, 2.5, 16, 25, 50, 75, 84, 97.5, 99.5, 100],
                )
            ]
        )
    )
    val_loss_stats = np.sqrt(val_loss_stats)
    val_stats = np.concatenate(
        [val_loss_stats, val_dataset_std * val_loss_stats]
    )
    
    stats = np.concatenate([train_stats, val_stats], axis=0)
    
    train_loss_vals = np.sqrt(train_loss_vals)
    val_loss_vals = np.sqrt(val_loss_vals)
    
    return stats, train_loss_vals, val_loss_vals