'''
Dataset classes for use in model training
'''

from typing import Union
import pickle

import numpy as np

from torch import Tensor, tensor
from torch import concatenate, unsqueeze
from torch.utils.data import Dataset


class EmitEcostressDataset(Dataset):
    def __init__(
        self,
        emit_data_path: Union[str, None] = None,
        emit_data: Union[np.ndarray, None] = None,
        omit_components: int = 0,
        ecostress_data_path: Union[str, None] = None,
        ecostress_data: Union[np.ndarray, None] = None,
        ecostress_center: Union[float, None] = None,
        ecostress_scale: Union[float, None] = None,
        additional_data_paths: Union[tuple[str], None] = None,
        additional_data: Union[tuple[np.ndarray], None] = None,
        device: str = 'cpu',
    ) -> None:
        '''
        Builds a pytorch dataset for predicting ECOSTRESS LST
        from EMIT spectra and any given additional data

        Input:
        emit_data_path: path to emit data as .npy or .pkl file,
                        not used if emit_data is not None
        emit_data: 2- or 3-dimensional np array of emit data
        omit_components: int telling how many components of the emit
                         spectra being loaded/provided to omit rom the end
        ecostress_data_path: path to ecostress data as .npy or .pkl file,
                             not used if ecostress_data is not None
        ecostress_data: 1- or 2-dimensional np array of ecostress data
        ecostress_center: float to center ecostress_data with
        ecostress_scale: float to scale centered ecostress_data by
        additional_data_paths: tuple of paths to supplementary datasets
                               used for additional model input,
                               must be .npy or .pkl files
        additional_data: tuple of 1-, 2-, or 3-dimensional np arrays of
                         supplementary data used for additional model input

        * Note that emit_data and ecostress_data will take precedence
          over emit_data_path and ecostress_data_path, respectively,
          but additional_data_paths and additional_data can both be specified,
          and if additional_data_paths specifies a path to data already
          in additional_data, a duplicate will be added
        '''
        if emit_data_path is not None and emit_data == None:
            if emit_data_path[-4:] == '.npy':
                emit_data = np.load(file=emit_data_path)
            elif emit_data_path[-4:] == '.pkl':
                emit_data = pickle.load(file=open(file=emit_data_path, mode='rb'))
            else:
                raise ValueError(
                    f'emit_data_path [{emit_data_path}] has an '
                    'invalid file extension, must be .npy or .pkl'
                )
        self.omit_components = omit_components
        if emit_data is not None:
            if len(emit_data.shape) == 2:
                self.emit_data = emit_data
                if self.omit_components > 0:
                    self.emit_data = self.emit_data[:, :-self.omit_components]
            elif len(emit_data.shape) == 3:
                self.emit_data = emit_data.reshape(
                    (
                        emit_data.shape[0] * emit_data.shape[1],
                        emit_data.shape[2]
                    )
                )
                if self.omit_components > 0:
                    self.emit_data = self.emit_data[:, :-self.omit_components]
            else:
                raise ValueError(
                    'emit_data must be 2- or 3-dimensional, '
                    f'found {len(emit_data.shape)}-dimensional'
                )
        else:
            raise ValueError(
                'Either emit_data_path or emit_data must not be None'
            )

        if ecostress_data_path is not None and ecostress_data == None:
            if ecostress_data_path[-4:] == '.npy':
                ecostress_data = np.load(file=ecostress_data_path)
            elif ecostress_data_path[-4:] == '.pkl':
                ecostress_data = pickle.load(file=open(file=ecostress_data_path, mode='rb'))
            else:
                raise ValueError(
                    f'ecostress_data_path [{ecostress_data_path}] has an '
                    'invalid file extension, must be .npy or .pkl'
                )
        if ecostress_data is not None:
            if len(ecostress_data.shape) == 1:
                self.ecostress_data = ecostress_data
            elif len(ecostress_data.shape) == 2:
                self.ecostress_data = ecostress_data.reshape(
                    ecostress_data.shape[0] * ecostress_data.shape[1]
                )
            else:
                raise ValueError(
                    'ecostress_data must be 1- or 2-dimensional, '
                    f'found {len(ecostress_data.shape)}-dimensional'
                )
        else:
            raise ValueError(
                'Either ecostress_data_path or ecostress_data must not be None'
            )

        assert self.emit_data.shape[0] == self.ecostress_data.shape[0], \
            'emit_data and ecostress_data must have the length, ' \
            f'got {self.emit_data.shape[0]} and {self.ecostress_data.shape[0]}'

        if ecostress_center is not None and ecostress_scale is not None:
            self.ecostress_center = ecostress_center
            self.ecostress_scale = ecostress_scale
        elif ecostress_center is None and ecostress_scale is None:
            self.ecostress_center = np.mean(self.ecostress_data, axis=0)
            self.ecostress_scale = np.std(a=self.ecostress_data, axis=0)
        elif ecostress_center is not None:
            self.ecostress_center = ecostress_center
            self.ecostress_scale = np.std(a=self.ecostress_data, axis=0)
        else:
            self.ecostress_center = np.mean(self.ecostress_data, axis=0)
            self.ecostress_scale = ecostress_scale
        
        self.ecostress_data = (
            (
                self.ecostress_data - self.ecostress_center
            ) / self.ecostress_scale
        )

        if np.sum(np.isnan(self.emit_data)) > 0:
            if np.sum(np.isnan(self.ecostress_data)) > 0:
                raise ValueError(
                    'No nan values are allowed: emit_data has '
                    f'{np.sum(np.isnan(self.emit_data))} nan values '
                    'and ecostress_data has '
                    f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
                )
            else:
                raise ValueError(
                    'No nan values are allowed: emit_data has '
                    f'{np.sum(np.isnan(self.emit_data))} nan values.'
                )
        elif np.sum(np.isnan(self.ecostress_data)) > 0:
            raise ValueError(
                'No nan values are allowed: ecostress_data has '
                f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
            )

        self.additional_data = []
        if additional_data is not None:
            for i in range(len(additional_data)):
                if len(additional_data[i].shape) == 1:
                    self.additional_data.append(additional_data[i])
                elif len(additional_data[i].shape) == 2:
                    self.additional_data.append(additional_data[i])
                elif len(additional_data[i].shape) == 3:
                    self.additional_data.append(
                        additional_data[i].reshape(
                            (
                                additional_data[i].shape[0] *
                                additional_data[i].shape[1],
                                additional_data[i].shape[2]
                            )
                        )
                    )
                else:
                    raise ValueError(
                        f'Item at index {i} in additional_data must be '
                        '1-, 2-, or 3-dimensional, '
                        f'found {len(additional_data[i].shape)}-dimensional'
                    )

        if additional_data_paths is not None:
            for i in range(len(additional_data_paths)):
                if additional_data_paths[i][-4:] == '.npy':
                    additional_data_element = np.load(file=additional_data_paths[i])
                elif additional_data_paths[i][-4:] == '.pkl':
                    additional_data_element = pickle.load(
                        file=open(file=additional_data_paths[i], mode='rb')
                    )
                else:
                    raise ValueError(
                        f'Path at index {i} [{additional_data_paths[i]}] '
                        'of additional_data_paths has an invalid '
                        'file extension, must be .npy or .pkl'
                    )
                if len(additional_data_element.shape) == 1:
                    self.additional_data.append(
                        additional_data_element.reshape(
                            (additional_data_element.shape[0], 1)
                        )
                    )
                if len(additional_data_element.shape) == 2:
                    self.additional_data.append(additional_data_element)
                elif len(additional_data_element.shape) == 3:
                    self.additional_data.append(
                        additional_data_element.reshape(
                            (
                                additional_data_element.shape[0] *
                                additional_data_element.shape[1],
                                additional_data_element.shape[2]
                            )
                        )
                    )
                else:
                    raise ValueError(
                        f'Item loaded from index {i} in additional_data_paths '
                        ' must be 1-, 2-, or 3-dimensional, found '
                        f'{len(additional_data_element.shape)}-dimensional'
                    )

        for i, additional_data_element in (
            enumerate(iterable=self.additional_data)
        ):
            if i >= len(self.additional_data):
                index: int = i - len(self.additional_data)
                message: str = f'loaded from index {index} in additional_data_paths'
            else:
                index = i
                message = f'from index {index} in additional_data'
            assert additional_data_element.shape[0] == self.emit_data.shape[0],\
                f'emit_data and additonal data {message} must have the same ' \
                f'length, got {self.emit_data.shape[0]} ' \
                f'and {additional_data_element.shape[0]}'

        self.input_dim: int = (
            self.emit_data.shape[1] +
            sum(
                [
                    additional_data_element.shape[1] if (
                        len(additional_data_element.shape) > 1
                    ) else 1
                    for additional_data_element in self.additional_data
                ]
            )
        )

        self.device = device
        self.emit_data = tensor(self.emit_data, device=self.device)
        self.ecostress_data = tensor(self.ecostress_data, device=self.device)
        for i, additional_data_element in enumerate(self.additional_data):
            self.additional_data[i] = tensor(
                additional_data_element, device=self.device
            )


    def __len__(self) -> int:
        return self.emit_data.shape[0]


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.emit_data[index, :]
        for additional_data_element in self.additional_data:
            if len(additional_data_element.shape) > 1:
                x = concatenate(
                    [x, additional_data_element[index, :]], dim=1
                )
            elif len(additional_data_element.shape) == 1:
                x = concatenate(
                    [x, unsqueeze(additional_data_element[index], 0)], dim=-1
                )
        y = self.ecostress_data[index]
        return x, y


class CNNDataset(Dataset):
    def __init__(
        self,
        emit_data_path: Union[str, None] = None,
        emit_data: Union[np.ndarray, None] = None,
        omit_components: int = 0,
        ecostress_data_path: Union[str, None] = None,
        ecostress_data: Union[np.ndarray, None] = None,
        ecostress_center: Union[float, None] = None,
        ecostress_scale: Union[float, None] = None,
        additional_data_paths: Union[tuple[str], None] = None,
        additional_data: Union[tuple[np.ndarray], None] = None,
        y_size: int = 50,
        x_size: int = 50,
    ) -> None:
        '''
        Builds a pytorch dataset for predicting ECOSTRESS LST
        from EMIT spectra and any given additional data

        Input:
        emit_data_path: path to emit data as .npy or .pkl file,
                        not used if emit_data is not None *
        emit_data: 3-dimensional np array of emit data
        omit_components: int telling how many components of the emit
                         spectra being loaded/provided to omit rom the end
        ecostress_data_path: path to ecostress data as .npy or .pkl file,
                             not used if ecostress_data is not None *
        ecostress_data: 2-dimensional np array of ecostress data
        ecostress_center: float to center ecostress_data with
        ecostress_scale: float to scale centered ecostress_data by
        additional_data_paths: tuple of paths to supplementary datasets
                               used for additional model input,
                               must be .npy or .pkl files
        additional_data: tuple of 2-, or 3-dimensional np arrays of
                         supplementary data used for additional model input
        y_size: number of pixels each training image is in the y-direction
        x_size: number of pixels each training image is in the x-direction

        * Note that emit_data and ecostress_data will take precedence
          over emit_data_path and ecostress_data_path, respectively,
          but additional_data_paths and additional_data can both be specified,
          and if additional_data_paths specifies a path to data already
          in additional_data, a duplicate will be added
        '''
        if emit_data_path is not None and emit_data == None:
            if emit_data_path[-4:] == '.npy':
                emit_data = np.load(file=emit_data_path)
            elif emit_data_path[-4:] == '.pkl':
                emit_data = pickle.load(file=open(file=emit_data_path, mode='rb'))
            else:
                raise ValueError(
                    f'emit_data_path [{emit_data_path}] has an '
                    'invalid file extension, must be .npy or .pkl'
                )
        self.omit_components = omit_components
        if emit_data is not None:
            if len(emit_data.shape) == 3:
                self.emit_data = emit_data
                if self.omit_components > 0:
                    self.emit_data = (
                        self.emit_data[:, :, :-self.omit_components]
                    )
            else:
                raise ValueError(
                    'emit_data must be 3-dimensional, '
                    f'found {len(emit_data.shape)}-dimensional'
                )
        else:
            raise ValueError(
                'Either emit_data_path or emit_data must not be None'
            )

        if ecostress_data_path is not None and ecostress_data == None:
            if ecostress_data_path[-4:] == '.npy':
                ecostress_data = np.load(file=ecostress_data_path)
            elif ecostress_data_path[-4:] == '.pkl':
                ecostress_data = pickle.load(file=open(file=ecostress_data_path, mode='rb'))
            else:
                raise ValueError(
                    f'ecostress_data_path [{ecostress_data_path}] has an '
                    'invalid file extension, must be .npy or .pkl'
                )
        if ecostress_data is not None:
            if len(ecostress_data.shape) == 2:
                self.ecostress_data = ecostress_data
            else:
                raise ValueError(
                    'ecostress_data must be 2-dimensional, '
                    f'found {len(ecostress_data.shape)}-dimensional'
                )
        else:
            raise ValueError(
                'Either ecostress_data_path or ecostress_data must not be None'
            )

        assert self.emit_data.shape[0] == self.ecostress_data.shape[0], \
            'emit_data and ecostress_data must have the same ' \
            f'first dimension, got shapes {self.emit_data.shape}' \
            f'and {self.ecostress_data.shape}'
        assert self.emit_data.shape[1] == self.ecostress_data.shape[1], \
            'emit_data and ecostress_data must have the same ' \
            f'second dimension, got shapes {self.emit_data.shape}' \
            f'and {self.ecostress_data.shape}'

        if ecostress_center is not None and ecostress_scale is not None:
            self.ecostress_center = ecostress_center
            self.ecostress_scale = ecostress_scale
        elif ecostress_center is None and ecostress_scale is None:
            self.ecostress_center = np.mean(self.ecostress_data)
            self.ecostress_scale = np.std(self.ecostress_data)
        elif ecostress_center is not None:
            self.ecostress_center = ecostress_center
            self.ecostress_scale = np.std(self.ecostress_data)
        else:
            self.ecostress_center = np.mean(self.ecostress_data)
            self.ecostress_scale = ecostress_scale

        self.ecostress_data = (
            (
                self.ecostress_data - self.ecostress_center
            ) / self.ecostress_scale
        )

        if np.sum(np.isnan(self.emit_data)) > 0:
            if np.sum(np.isnan(self.ecostress_data)) > 0:
                raise ValueError(
                    'No nan values are allowed: emit_data has '
                    f'{np.sum(np.isnan(self.emit_data))} nan values '
                    'and ecostress_data has '
                    f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
                )
            else:
                raise ValueError(
                    'No nan values are allowed: emit_data has '
                    f'{np.sum(np.isnan(self.emit_data))} nan values.'
                )
        elif np.sum(np.isnan(self.ecostress_data)) > 0:
            raise ValueError(
                'No nan values are allowed: ecostress_data has '
                f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
            )

        self.additional_data = []
        if additional_data is not None:
            for i in range(len(additional_data)):
                if len(additional_data[i].shape) == 2:
                    self.additional_data.append(
                        additional_data[i].reshape(
                            additional_data[i].shape[0],
                            additional_data[i].shape[1],
                            1,
                        )
                    )
                elif len(additional_data[i].shape) == 3:
                    self.additional_data.append(additional_data[i])
                else:
                    raise ValueError(
                        f'Item at index {i} in additional_data must be '
                        '2-, or 3-dimensional, '
                        f'found {len(additional_data[i].shape)}-dimensional'
                    )

        if additional_data_paths is not None:
            for i in range(len(additional_data_paths)):
                if additional_data_paths[i][-4:] == '.npy':
                    additional_data_element = np.load(file=additional_data_paths[i])
                elif additional_data_paths[i][-4:] == '.pkl':
                    additional_data_element = pickle.load(
                        file=open(file=additional_data_paths[i], mode='rb')
                    )
                else:
                    raise ValueError(
                        f'Path at index {i} [{additional_data_paths[i]}] '
                        'of additional_data_paths has an invalid '
                        'file extension, must be .npy or .pkl'
                    )
                if len(additional_data_element.shape) == 2:
                    self.additional_data.append(
                        additional_data_element.reshape(
                            additional_data_element.shape[0],
                            additional_data_element.shape[1],
                            1,
                        )
                    )
                elif len(additional_data_element.shape) == 3:
                    self.additional_data.append(additional_data_element)
                else:
                    raise ValueError(
                        f'Item loaded from index {i} in additional_data_paths '
                        ' must be 2-, or 3-dimensional, found '
                        f'{len(additional_data_element.shape)}-dimensional'
                    )

        for i, additional_data_element in (
            enumerate(iterable=self.additional_data)
        ):
            if i >= len(self.additional_data):
                index: int = i - len(self.additional_data)
                message: str = f'loaded from index {index} in additional_data_paths'
            else:
                index = i
                message = f'from index {index} in additional_data'
            assert additional_data_element.shape[0] == self.emit_data.shape[0],\
                f'emit_data and additonal data {message} must have the same ' \
                f'first dimension, got shapes {self.emit_data.shape} ' \
                f'and {additional_data_element.shape}'
            assert additional_data_element.shape[1] == self.emit_data.shape[1],\
                f'emit_data and additonal data {message} must have the same ' \
                f'second dimension, got shapes {self.emit_data.shape} ' \
                f'and {additional_data_element.shape}'

        self.input_dim: int = (
            self.emit_data.shape[2] +
            sum(
                [
                    additional_data_element.shape[2] if (
                        len(additional_data_element.shape) == 3
                    ) else 1
                    for additional_data_element in self.additional_data
                ]
            )
        )
        
        if emit_data.shape[0] % y_size != 0:
            if emit_data.shape[1] % y_size != 0:
                raise ValueError(
                    'y_size must divide length of emit first dimension, ' \
                    f'got y_size {y_size} and length {emit_data.shape[0]}, ' \
                    'and x_size must divide length of emit second dimension, ' \
                    f'got x_size {x_size} and length {emit_data.shape[1]}.'
                )
            else:
                raise ValueError(
                    'y_size must divide length of emit first dimension, ' \
                    f'got y_size {y_size} and length {emit_data.shape[0]}.'
                )
        else:
            if emit_data.shape[1] % x_size != 0:
                raise ValueError(
                    'x_size must divide length of emit second dimension, ' \
                    f'got x_size {x_size} and length {emit_data.shape[1]}.'
                )
        
        self.x_size = x_size
        self.y_size = y_size
        
        self.n_images_y = emit_data.shape[0] // self.y_size
        self.n_images_x = emit_data.shape[1] // self.x_size
        
        self.emit_data = tensor(self.emit_data)
        self.ecostress_data = tensor(self.ecostress_data)
        for i, additional_data_element in enumerate(self.additional_data):
            self.additional_data[i] = tensor(additional_data_element)


    def __len__(self) -> int:
        return self.n_images_y * self.n_images_x


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = index // self.n_images_x
        column = index % self.n_images_x
        
        x = self.emit_data[
            row * self.y_size : (row + 1) * self.y_size,
            column * self.x_size : (column + 1) * self.x_size,
            :,
        ]
        
        for additional_data_element in self.additional_data:
            x = concatenate(
                [
                    x,
                    additional_data_element[
                        row * self.y_size : (row + 1) * self.y_size,
                        column * self.x_size : (column + 1) * self.x_size,
                        :,
                    ]
                ],
                dim=2,
            )

        y = self.ecostress_data[
            row * self.y_size : (row + 1) * self.y_size,
            column * self.x_size : (column + 1) * self.x_size,
        ]
                
        return x, y
    
    
    
class PatchToPixelDataset(Dataset):
    def __init__(
        self,
        emit_data_path: Union[str, None] = None,
        emit_data: Union[np.ndarray, None] = None,
        omit_components: int = 0,
        ecostress_data_path: Union[str, None] = None,
        ecostress_data: Union[np.ndarray, None] = None,
        ecostress_center: Union[float, None] = None,
        ecostress_scale: Union[float, None] = None,
        additional_data_paths: Union[tuple[str], None] = None,
        additional_data: Union[tuple[np.ndarray], None] = None,
        radius: int = 3,
        boundary_width: int = 16,
    ) -> None:
        '''
        Builds a pytorch dataset for predicting ECOSTRESS LST
        from EMIT spectra and any given additional data

        Input:
        emit_data_path: path to emit data as .npy or .pkl file,
                        not used if emit_data is not None *
        emit_data: 3-dimensional np array of emit data
        omit_components: int telling how many components of the emit
                         spectra being loaded/provided to omit rom the end
        ecostress_data_path: path to ecostress data as .npy or .pkl file,
                             not used if ecostress_data is not None *
        ecostress_data: 2-dimensional np array of ecostress data
        ecostress_center: float to center ecostress_data with
        ecostress_scale: float to scale centered ecostress_data by
        additional_data_paths: tuple of paths to supplementary datasets
                               used for additional model input,
                               must be .npy or .pkl files
        additional_data: tuple of 2-, or 3-dimensional np arrays of
                         supplementary data used for additional model input
        radius: number of pixels in each direction around the center pixel
                to include for each training point, i.e., the block of input
                is of shape ((2 * radius) + 1) x ((2 * radius) + 1) x depth
        boundary_width: number of pixels on each edge of the dataset to ignore
                        as part of the indices of the dataset that get used

        * Note that emit_data and ecostress_data will take precedence
          over emit_data_path and ecostress_data_path, respectively,
          but additional_data_paths and additional_data can both be specified,
          and if additional_data_paths specifies a path to data already
          in additional_data, a duplicate will be added
          
        
        Yields on call to __getitem__():
        x: (((2 * radius) + 1) * ((2 * radius) + 1))
           by
           (emit_length + 1 + additional_data_lengths)
           tensor containing emit spectra for a block of data with
           the specified radius around the pixel of interest
        y: float that is the number of standard deviations away the
           temperature at the pixel of interest is from the mean temperature
        '''
        if emit_data_path is not None and emit_data == None:
            if emit_data_path[-4:] == '.npy':
                emit_data = np.load(file=emit_data_path)
            elif emit_data_path[-4:] == '.pkl':
                emit_data = pickle.load(file=open(file=emit_data_path, mode='rb'))
            else:
                raise ValueError(
                    f'emit_data_path [{emit_data_path}] has an '
                    'invalid file extension, must be .npy or .pkl'
                )
        self.omit_components = omit_components
        if emit_data is not None:
            if len(emit_data.shape) == 3:
                self.emit_data = emit_data
                if self.omit_components > 0:
                    self.emit_data = (
                        self.emit_data[:, :, :-self.omit_components]
                    )
            else:
                raise ValueError(
                    'emit_data must be 3-dimensional, '
                    f'found {len(emit_data.shape)}-dimensional'
                )
        else:
            raise ValueError(
                'Either emit_data_path or emit_data must not be None'
            )

        if ecostress_data_path is not None and ecostress_data == None:
            if ecostress_data_path[-4:] == '.npy':
                ecostress_data = np.load(file=ecostress_data_path)
            elif ecostress_data_path[-4:] == '.pkl':
                ecostress_data = pickle.load(file=open(file=ecostress_data_path, mode='rb'))
            else:
                raise ValueError(
                    f'ecostress_data_path [{ecostress_data_path}] has an '
                    'invalid file extension, must be .npy or .pkl'
                )
        if ecostress_data is not None:
            if len(ecostress_data.shape) == 2:
                self.ecostress_data = ecostress_data
            else:
                raise ValueError(
                    'ecostress_data must be 2-dimensional, '
                    f'found {len(ecostress_data.shape)}-dimensional'
                )
        else:
            raise ValueError(
                'Either ecostress_data_path or ecostress_data must not be None'
            )

        assert self.emit_data.shape[0] == self.ecostress_data.shape[0], \
            'emit_data and ecostress_data must have the same ' \
            f'first dimension, got shapes {self.emit_data.shape}' \
            f'and {self.ecostress_data.shape}'
        assert self.emit_data.shape[1] == self.ecostress_data.shape[1], \
            'emit_data and ecostress_data must have the same ' \
            f'second dimension, got shapes {self.emit_data.shape}' \
            f'and {self.ecostress_data.shape}'

        if ecostress_center is not None and ecostress_scale is not None:
            self.ecostress_center = ecostress_center
            self.ecostress_scale = ecostress_scale
        elif ecostress_center is None and ecostress_scale is None:
            self.ecostress_center = np.mean(self.ecostress_data)
            self.ecostress_scale = np.std(self.ecostress_data)
        elif ecostress_center is not None:
            self.ecostress_center = ecostress_center
            self.ecostress_scale = np.std(self.ecostress_data)
        else:
            self.ecostress_center = np.mean(self.ecostress_data)
            self.ecostress_scale = ecostress_scale

        self.ecostress_data = (
            (
                self.ecostress_data - self.ecostress_center
            ) / self.ecostress_scale
        )

        if np.sum(np.isnan(self.emit_data)) > 0:
            if np.sum(np.isnan(self.ecostress_data)) > 0:
                raise ValueError(
                    'No nan values are allowed: emit_data has '
                    f'{np.sum(np.isnan(self.emit_data))} nan values '
                    'and ecostress_data has '
                    f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
                )
            else:
                raise ValueError(
                    'No nan values are allowed: emit_data has '
                    f'{np.sum(np.isnan(self.emit_data))} nan values.'
                )
        elif np.sum(np.isnan(self.ecostress_data)) > 0:
            raise ValueError(
                'No nan values are allowed: ecostress_data has '
                f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
            )

        self.additional_data = []
        if additional_data is not None:
            for i in range(len(additional_data)):
                if len(additional_data[i].shape) == 2:
                    self.additional_data.append(
                        additional_data[i].reshape(
                            additional_data[i].shape[0],
                            additional_data[i].shape[1],
                            1,
                        )
                    )
                elif len(additional_data[i].shape) == 3:
                    self.additional_data.append(additional_data[i])
                else:
                    raise ValueError(
                        f'Item at index {i} in additional_data must be '
                        '2-, or 3-dimensional, '
                        f'found {len(additional_data[i].shape)}-dimensional'
                    )

        if additional_data_paths is not None:
            for i in range(len(additional_data_paths)):
                if additional_data_paths[i][-4:] == '.npy':
                    additional_data_element = np.load(file=additional_data_paths[i])
                elif additional_data_paths[i][-4:] == '.pkl':
                    additional_data_element = pickle.load(
                        file=open(file=additional_data_paths[i], mode='rb')
                    )
                else:
                    raise ValueError(
                        f'Path at index {i} [{additional_data_paths[i]}] '
                        'of additional_data_paths has an invalid '
                        'file extension, must be .npy or .pkl'
                    )
                if len(additional_data_element.shape) == 2:
                    self.additional_data.append(
                        additional_data_element.reshape(
                            additional_data_element.shape[0],
                            additional_data_element.shape[1],
                            1,
                        )
                    )
                elif len(additional_data_element.shape) == 3:
                    self.additional_data.append(additional_data_element)
                else:
                    raise ValueError(
                        f'Item loaded from index {i} in additional_data_paths '
                        ' must be 2-, or 3-dimensional, found '
                        f'{len(additional_data_element.shape)}-dimensional'
                    )

        for i, additional_data_element in (
            enumerate(iterable=self.additional_data)
        ):
            if i >= len(self.additional_data):
                index = i - len(self.additional_data)
                message = f'loaded from index {index} in additional_data_paths'
            else:
                index = i
                message = f'from index {index} in additional_data'
            assert additional_data_element.shape[0] == self.emit_data.shape[0],\
                f'emit_data and additonal data {message} must have the same ' \
                f'first dimension, got shapes {self.emit_data.shape} ' \
                f'and {additional_data_element.shape}'
            assert additional_data_element.shape[1] == self.emit_data.shape[1],\
                f'emit_data and additonal data {message} must have the same ' \
                f'second dimension, got shapes {self.emit_data.shape} ' \
                f'and {additional_data_element.shape}'

        self.input_dim: int = (
            self.emit_data.shape[2] +
            # removing positional encoding since it is unneccesary
            # 1 +  # from location encoding d defined below
            sum(
                [
                    additional_data_element.shape[2] if (
                        len(additional_data_element.shape) == 3
                    ) else 1
                    for additional_data_element in self.additional_data
                ]
            )
        )
        
        self.radius = radius
        self.boundary_width = boundary_width
        
        # removing positional encoding since it is unneccesary
        # self.d = np.array(
        #     [
        #         [
        #                 (i**2 + j**2)**(1/2)
        #                 for j in range(-self.radius, self.radius + 1)
        #         ] for i in range(-self.radius, self.radius + 1)
        #     ]
        # )
        
        self.emit_data = tensor(self.emit_data)
        self.ecostress_data = tensor(self.ecostress_data)
        for i, additional_data_element in enumerate(self.additional_data):
            self.additional_data[i] = tensor(additional_data_element)
        # removing positional encoding since it is unneccesary
        # self.d = tensor(self.d)


    def __len__(self) -> int:
        return (
            len(self.ecostress_data.flatten()) - 
            2 * self.boundary_width * (
                self.ecostress_data.shape[0] + 
                self.ecostress_data.shape[1] - 
                2 * self.boundary_width
            )
        )


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        real_index = (
            index +
            self.boundary_width * self.ecostress_data.shape[1] +
            self.boundary_width +
            2 * self.boundary_width * (
                index // (
                    self.ecostress_data.shape[1] - 2 * self.boundary_width
                )
            )
        )
        
        return_indices = np.array(
            [
                [
                        real_index + j + (self.ecostress_data.shape[1] * i)
                        for j in range(-self.radius, self.radius + 1)
                ] for i in range(-self.radius, self.radius + 1)
            ],
            dtype=int,
        ).flatten()
                
        return_indices = np.unravel_index(
            return_indices, self.ecostress_data.shape
        )
        
        x = self.emit_data[return_indices].reshape(
            (2 * self.radius + 1, 2 * self.radius + 1, self.emit_data.shape[2])
        )
        
        # removing positional encoding as it is unneccesary
        # x = concatenate(
        #     [x, unsqueeze(self.d, dim=-1)], dim=-1
        # )
        
        for additional_data_element in self.additional_data:
            x = concatenate(
                [
                    x,
                    additional_data_element[return_indices].reshape(
                        (2 * self.radius + 1, 2 * self.radius + 1, 1)
                    )
                ],
                dim=-1,
            )
            
        x = x.flatten()
        
        y = self.ecostress_data[
            np.unravel_index(real_index, self.ecostress_data.shape)
        ]
                
        return x, y