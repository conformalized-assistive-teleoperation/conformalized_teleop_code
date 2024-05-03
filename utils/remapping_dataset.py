"""
la_dataset.py

Core class definition for datasets and preprocessing, and other necessary startup steps to run before training the
vanilla auto-encoding (Latent Actions) models.
"""


import torch
from torch.utils.data import ConcatDataset, Dataset


class RemappedDataset(Dataset):
    """ Dataset Wrapper around a Single Demonstration in the Dataset. """

    def __init__(
        self, full_remapped_dataset
    ):
        self.full_remapped_dataset = full_remapped_dataset


    def __len__(self) -> int:
        return len(self.full_remapped_dataset)

    def __getitem__(self, idx: int):
        human_input_state, predicted_action, next_state_applying_action, rotated_z = self.full_remapped_dataset[idx]

        # convet all to tensors
        human_input_state = torch.FloatTensor(human_input_state[0:7])
        predicted_action = torch.FloatTensor(predicted_action[0:7])
        rotated_z = torch.FloatTensor(rotated_z)

        return human_input_state, rotated_z, predicted_action

class Nav2DRemappedDataset(Dataset):
    """ Dataset Wrapper around a Single Demonstration in the Dataset. """

    def __init__(
        self, full_remapped_dataset
    ):
        self.full_remapped_dataset = full_remapped_dataset


    def __len__(self) -> int:
        return len(self.full_remapped_dataset)

    def __getitem__(self, idx: int):
        human_input_state, predicted_action, rotated_z = self.full_remapped_dataset[idx]

        # convet all to tensors
        human_input_state = torch.FloatTensor(human_input_state)
        predicted_action = torch.FloatTensor(predicted_action)
        rotated_z = torch.FloatTensor([rotated_z])

        return human_input_state, rotated_z, predicted_action
