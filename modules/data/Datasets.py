from typing import Union
import pickle

import numpy as np

from torch import Tensor, tensor
from torch import concatenate
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
                        not used if emit_data != None
        emit_data: 2- or 3- dimensional np array of emit data
        omit_components: int telling how many components of the emit
                         spectra being loaded/provided to omit rom the end
        ecostress_data_path: path to ecostress data as .npy or .pkl file,
                             not used if ecostress_data != None
        ecostress_data: 1- or 2- dimensional np array of ecostress data
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
        if emit_data_path != None and emit_data == None:
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

        if ecostress_data_path != None and ecostress_data == None:
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
                    'ecostress_data must be 2- or 3-dimensional, '
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
            raise ValueError(
                'Both center and scale must be provided, but scale was None'
            )
        else:
            raise ValueError(
                'Both center and scale must be provided, but center was None'
            )
        self.ecostress_data = (
            (
                self.ecostress_data - self.ecostress_center
            ) / self.ecostress_scale
        )

        if np.sum(np.isnan(self.emit_data)) > 0:
            if np.sum(np.isnan(self.ecostress_data)) > 0:
                raise ValueError(
                    'No nan values are allowed: emit_data has'
                    f'{np.sum(np.isnan(self.emit_data))} nan values '
                    'and ecostress_data has '
                    f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
                )
            else:
                raise ValueError(
                    'No nan values are allowed: emit_data has'
                    f'{np.sum(np.isnan(self.emit_data))} nan values.'
                )
        elif np.sum(np.isnan(self.ecostress_data)) > 0:
            raise ValueError(
                'No nan values are allowed: ecostress_data has'
                f'{np.sum(np.isnan(self.ecostress_data))} nan values.'
            )
        # nan_mask = np.isnan(self.emit_data)
        # self.emit_data = self.emit_data[~nan_mask]
        # self.ecostress_data = self.ecostress_data[~nan_mask[:,0]]

        # assert self.emit_data.shape[0] == self.ecostress_data.shape[0], \
        #     'emit_data and ecostress_data changed lengths during filtering, ' \
        #     f'got {self.emit_data.shape[0]} and {self.ecostress_data.shape[0]}'

        self.additional_data = []
        if additional_data != None:
            for i in range(len(additional_data)):
                if len(additional_data[i].shape) == 1:
                    self.additional_data.append(additional_data[i])
                if len(additional_data[i].shape) == 2:
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
                        f'Item at index {i} in additional_data must be'
                        '1-, 2-, or 3-dimensional, '
                        f'found {len(additional_data[i].shape)}-dimensional'
                    )

        if additional_data_paths != None:
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
                        ' must be 1-. 2-, or 3-dimensional, found '
                        f'{len(additional_data_element.shape)}-dimensional'
                    )

        for i, additional_data_element in (
            enumerate(iterable=self.additional_data)
        ):
            # additional_data_element = additional_data_element[~nan_mask]
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
                    additional_data_element.shape[1]
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
            x = concatenate(
                [x, additional_data_element[index, :]], dim=1 # type: ignore
            )
        y = self.ecostress_data[index]
        return x, y # type: ignore