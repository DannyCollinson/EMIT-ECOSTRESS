import numpy as np
import pandas as pd
from torch import Size


def initialize_eval_results() -> pd.DataFrame:
    '''
    Initializes the dataframe for the evaluation results
    '''
    return pd.DataFrame(
        columns=[
            'radius',
            'n_components',
            
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