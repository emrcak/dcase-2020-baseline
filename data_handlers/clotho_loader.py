#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, MutableMapping, Union,\
    Tuple, List
from pathlib import Path

from torch.utils.data import DataLoader
from torch import cat, zeros, from_numpy, ones, Tensor
from numpy import ndarray, unique
import pickle

from data_handlers._clotho import ClothoDataset

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_clotho_loader']


def _get_nb_output_classes(settings_data, settings_io):
    """Gets the amount of output classes.

    :param settings: Settings to use.
    :type settings: dict
    :return: Amount of output classes.
    :rtype: int
    """
    f_name_field = 'words_list_file_name' \
        if settings_data['output_field_name'].startswith('words') \
        else 'characters_list_file_name'

    f_name = settings_io['dataset']['files'][f_name_field]
    path = Path(
        settings_io['root_dirs']['data'],
        settings_io['dataset']['pickle_files_dir'],
        f_name)

    with path.open('rb') as f:
        return len(pickle.load(f))

class ClothoCollator(object):
    def __init__(self, settings_data, settings_io):
        self.nb_output_classes = _get_nb_output_classes(settings_data, settings_io)

    def __call__(self, batch):
        """Pads data.

        For each batch, the maximum input and output\
        time-steps are calculated. Then, then input and\
        output data are padded to match the maximum time-steps.

        The input data are padded with zeros in front, and\
        the output with] <EOS> tokens at the end.

        :param batch: Batch data of batch x time x features.\
                      First element in the list are the input\
                      data, second the output data.
        :type batch: list[numpy.ndarray]
        :return: Padded data. First tensor is the input data\
                 and second the output.
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor, list[str]
        """
        max_input_t_steps = max([i[0].shape[0] for i in batch])
        max_output_t_steps = max([i[1].shape[0] for i in batch])

        file_names = [i[3] for i in batch]

        input_features = batch[0][0].shape[-1]
        eos_token = batch[0][1][-1]

        input_tensor = cat([
            cat([from_numpy(i[0]).float(), zeros(
                max_input_t_steps - i[0].shape[0],
                input_features).float()]).unsqueeze(0) for i in batch])

        output_tensor = cat([
            cat([
                from_numpy(i[1]).long(),
                ones(max_output_t_steps - len(i[1])).mul(eos_token).long()
            ]).unsqueeze(0) for i in batch])

        output_keyword_tensor = cat([zeros(1, self.nb_output_classes).scatter_(1, from_numpy(unique(i[2])).long().unsqueeze(0), 1)
                                     for i in batch])

        return input_tensor, output_tensor, output_keyword_tensor, file_names


def get_clotho_loader(split: str,
                      is_training: bool,
                      settings_data: MutableMapping[
                          str, Union[str, bool, MutableMapping[str, str]]],
                      settings_io: MutableMapping[
                          str, Union[str, bool, MutableMapping[
                              str, Union[str, MutableMapping[str, str]]]]]) \
        -> DataLoader:
    """Gets the data loader.

    :param split: Split to be used.
    :type split: str
    :param is_training: Is training data?
    :type is_training: bool
    :param settings_data: Data loading and dataset settings.
    :type settings_data: dict
    :param settings_io: Files I/O settings.
    :type settings_io: dict
    :return: Data loader.
    :rtype: torch.utils.data.DataLoader
    """
    data_dir = Path(
        settings_io['root_dirs']['data'],
        settings_io['dataset']['features_dirs']['output'])

    dataset = ClothoDataset(
        data_dir=data_dir,
        split=split,
        input_field_name=settings_data['input_field_name'],
        output_field_name=settings_data['output_field_name'],
        output_keyword_field_name=settings_data['output_keyword_field_name'],
        load_into_memory=settings_data['load_into_memory'],
        sort_per_target_length=settings_data.get('sort_per_target_length', False))

    shuffle = settings_data['shuffle'] if is_training else False
    drop_last = settings_data['drop_last'] if is_training else False

    if shuffle and settings_data.get('sort_per_target_length', False):
        raise("Shuffling the data after sorting per target length is probably unintended. Please comment this out"
              "if it is not the case.")

    clotho_collator = ClothoCollator(settings_data, settings_io)

    return DataLoader(
        dataset=dataset,
        batch_size=settings_data['batch_size'],
        shuffle=shuffle,
        num_workers=settings_data['num_workers'],
        drop_last=drop_last,
        collate_fn=clotho_collator)

# EOF
