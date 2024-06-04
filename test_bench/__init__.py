import numpy as np
import tensorflow as tf

from src.datasets.utils_rich import get_merged_typed_dataset, parse_dataset_np


def get_checkpoint_name(particle: str = 'pion', dropout: str = '0.01', dropout_type: str = 'bernoulli_structured'):
    return f'{dropout_type}_dropout_line_test_cramer_drop_rate_{dropout}_{particle}'


def _split_by_line(df, slope=1, intercept=0):
    top_half = df[df['Brunel_ETA'] > df['Brunel_P'] * slope + intercept]
    bottom_half = df[df['Brunel_ETA'] <= df['Brunel_P'] * slope + intercept]

    top_half = top_half.reset_index(drop=True)
    bottom_half = bottom_half.reset_index(drop=True)

    return top_half, bottom_half


def split_by_line(df_train, df_test):
    return _split_by_line(df_train)[0], _split_by_line(df_test)[1]


def load_particle_datasets(particle, data_dir='../data/rich/'):
    """ The returned dictionary has this format:
        {
          "<particle_name>": {
            'data_train': data_train,
            'data_val': data_val,
            'scaler': scaler,
            'feats_train': feats_train,
            'targets_train': targets_train,
            'feats_val': feats_val,
            'targets_val': targets_val
          }
        }
    """
    data_train, data_val, scaler = get_merged_typed_dataset(data_dir, particle, dtype=np.float32, log=True,
                                                            sample_fn=split_by_line)
    feats_train, targets_train, _ = parse_dataset_np(data_train)
    feats_val, targets_val, _ = parse_dataset_np(data_val)

    print(f'feats_train shape\t{feats_train.shape}\n'
          f'targets_train shape\t{targets_train.shape}\n'
          f'feats_val shape  \t{feats_val.shape}\n'
          f'targets_val shape\t{targets_val.shape}\n')

    return {
        'data_train': data_train,
        'data_val': data_val,
        'scaler': scaler,
        'feats_train': feats_train,
        'targets_train': targets_train,
        'feats_val': feats_val,
        'targets_val': targets_val
    }


def subsample_dataset(features_ds, targets_ds, subsample_percent_size=1.0, debug=False):
    n = int(len(features_ds) * subsample_percent_size)
    sample_idxs = np.random.choice(np.arange(len(features_ds)), n, replace=False)
    x_sample = tf.constant(features_ds[sample_idxs])
    y_sample = tf.constant(targets_ds[sample_idxs])
    if debug:
        print(f'x_sample type = {type(x_sample)}\tshape={x_sample.shape}\n'
              f'  Brunel_P     Brunel_ETA  nTracks_Brunel\n{x_sample[:3]}\n')
        print(f'y_sample type = {type(y_sample)}\tshape={y_sample.shape}\n'
              f'  RichDLLe     RichDLLk    RichDLLmu   RichDLLp    RichDLLbt\n{y_sample[:3]}')

    return x_sample, y_sample
