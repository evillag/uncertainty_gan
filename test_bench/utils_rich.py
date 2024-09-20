"""
    The code in this file uses a codebase from 
    https://github.com/yandexdataschool/RICH-GAN/
"""

import glob
import os
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
raw_feature_columns = ['Brunel_P', 'Brunel_ETA', 'nTracks_Brunel']
weight_col = 'probe_sWeight'

y_count = len(dll_columns)

particle_types = ['kaon', 'pion', 'proton', 'muon']


def path_selection(particle, paths):
    return [path for path in paths if particle in path]


def parse_file_lists(data_dir):
    paths = glob.glob(os.path.join(data_dir, '*.csv'))

    file_lists = {particle: path_selection(particle, paths)
                  for particle in particle_types}

    return file_lists


def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
    return data[dll_columns + raw_feature_columns + [weight_col]]


def load_and_merge_and_cut(filename_list):
    return pd.concat(
        [load_and_cut(fname) for fname in filename_list],
        axis=0, ignore_index=True)


def split(data, test_size):
    data_train, data_val = train_test_split(
        data, test_size=test_size, random_state=42)
    data_val, data_test = train_test_split(
        data_val, test_size=test_size, random_state=1812)
    return data_train.reset_index(drop=True), \
        data_val.reset_index(drop=True), \
        data_test.reset_index(drop=True)


def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)


def get_merged_typed_dataset(
        data_dir,
        particle_type,
        test_size=0.5,
        dtype=None,
        log=False,
        n_quantiles=100000,
        sample_fn=None
):
    file_lists = parse_file_lists(data_dir)

    file_list = file_lists.get(particle_type, None)

    if file_list is None:
        raise KeyError(
            'particle_type should be in {}'.format(file_lists.keys()))

    if log:
        print("Reading and concatenating datasets:")
        for fname in file_list:
            print("\t{}".format(fname))

    data_full = load_and_merge_and_cut(file_list)

    # Must split the whole to preserve train/test split""
    if log:
        print("splitting to train/val/test")

    # split_fn = split_fn if split_fn is not None else split
    data_train, data_val, _ = split(data_full, test_size)

    data_train_orig = data_train.copy()
    data_val_orig = data_val.copy()

    if log:
        print("fitting the scaler")

    print("scaler train sample size: {}".format(len(data_train)))
    start_time = time()
    if n_quantiles == 0:
        scaler = StandardScaler().fit(data_train.drop(weight_col, axis=1).values)
    else:
        scaler = QuantileTransformer(output_distribution="normal",
                                     n_quantiles=n_quantiles,
                                     subsample=int(1e10)).fit(data_train.drop(weight_col, axis=1).values)
    print("scaler n_quantiles: {}, time = {}".format(
        n_quantiles, time() - start_time))
    if log:
        print("scaling train set")

    data_train = pd.concat([scale_pandas(data_train.drop(
        weight_col, axis=1), scaler), data_train[weight_col]], axis=1)

    if log:
        print("scaling test set")

    data_val = pd.concat([scale_pandas(data_val.drop(
        weight_col, axis=1), scaler), data_val[weight_col]], axis=1)

    if dtype is not None:
        if log:
            print("converting dtype to {}".format(dtype))

        data_train = data_train.astype(dtype, copy=False)
        data_train_orig = data_train_orig.astype(dtype, copy=False)
        data_val = data_val.astype(dtype, copy=False)
        data_val_orig = data_val_orig.astype(dtype, copy=False)

    if sample_fn is not None:
        data_train, data_train_orig, data_val, data_val_orig = sample_fn(data_train, data_train_orig, data_val,
                                                                         data_val_orig)

    return data_train, data_val, scaler, data_train_orig, data_val_orig


def get_all_particles_dataset(data_dir, test_size=0.5, dtype=None, log=False, n_quantiles=100000):
    data_train_all = []
    data_val_all = []
    scaler_all = {}
    for index, particle in enumerate(particle_types):
        data_train, data_val, scaler, _, _ = get_merged_typed_dataset(
            data_dir, particle, test_size=test_size,
            dtype=dtype, log=log, n_quantiles=n_quantiles)

        ohe_table = pd.DataFrame(np.zeros((len(data_train), len(particle_types))),
                                 columns=['is_{}'.format(i) for i in particle_types])
        ohe_table['is_{}'.format(particle)] = 1

        data_train_all.append(pd.concat([data_train.iloc[:, :y_count],
                                         ohe_table,
                                         data_train.iloc[:, y_count:]], axis=1))

        data_val_all.append(pd.concat([data_val.iloc[:, :y_count],
                                       ohe_table[:len(data_val)].copy(),
                                       data_val.iloc[:, y_count:]], axis=1))
        scaler_all[index] = scaler
    data_train_all = pd.concat(
        data_train_all, axis=0).astype(dtype, copy=False)
    data_val_all = pd.concat(data_val_all, axis=0).astype(dtype, copy=False)
    return data_train_all, data_val_all, scaler_all


def parse_example(row):
    targets, features, weight = row[:5], row[-4:-1], row[-1]

    return features, targets, weight


def parse_dataset_np(data):
    targets = data.iloc[:, :5].to_numpy()
    features = data.iloc[:, -4:-1].to_numpy()
    weight = data.iloc[:, -1].to_numpy()

    return features, targets, weight
