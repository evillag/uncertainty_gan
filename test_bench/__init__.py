import numpy as np
import tensorflow as tf

from .utils_rich import get_merged_typed_dataset, parse_dataset_np


def get_checkpoint_name(particle: str = 'pion', dropout: str = '0.01', dropout_type: str = 'bernoulli_structured'):
    return f'{dropout_type}_dropout_line_test_cramer_drop_rate_{dropout}_{particle}'


def _split_by_line(df, df2, slope=1, intercept=0):
    top_half = df[df['Brunel_ETA'] > df['Brunel_P'] * slope + intercept]
    bottom_half = df[df['Brunel_ETA'] <= df['Brunel_P'] * slope + intercept]


    top_half_indices = top_half.index
    bottom_half_indices = bottom_half.index

    top_half = top_half.reset_index(drop=True)
    bottom_half = bottom_half.reset_index(drop=True)

    top_half_2 = df2.loc[top_half_indices].reset_index(drop=True)
    bottom_half_2 = df2.loc[bottom_half_indices].reset_index(drop=True)

    return top_half, top_half_2, bottom_half, bottom_half_2


def split_by_line(df_train, df_train_orig, df_test, df_test_orig):
    top_half, top_half_2, _, _ = _split_by_line(df_train, df_train_orig)
    _, _, bottom_half, bottom_half_2 =_split_by_line(df_test, df_test_orig)
    return top_half, top_half_2, bottom_half, bottom_half_2


def load_particle_datasets(particle, data_dir='../data/rich/'):
    """ The returned dictionary has this format:
        {
            'data_train': data_train,
            'data_val': data_val,
            'scaler': scaler,
            'feats_train': feats_train,
            'targets_train': targets_train,
            'feats_val': feats_val,
            'targets_val': targets_val,
            'feats_train_orig': feats_train_orig,
            'targets_train_orig': targets_train_orig,
            'feats_val_orig': feats_val_orig,
            'targets_val_orig': targets_val_orig,
        }
    """
    data_train, data_val, scaler, data_train_orig, data_val_orig,  = get_merged_typed_dataset(data_dir, particle, dtype=np.float32, log=True,
                                                            sample_fn=split_by_line)
    feats_train, targets_train, _ = parse_dataset_np(data_train)
    feats_val, targets_val, _ = parse_dataset_np(data_val)

    feats_train_orig, targets_train_orig, _ = parse_dataset_np(data_train_orig)
    feats_val_orig, targets_val_orig, _ = parse_dataset_np(data_val_orig)

    print(f'feats_train shape\t{feats_train.shape}\n'
          f'targets_train shape\t{targets_train.shape}\n'
          f'feats_val shape  \t{feats_val.shape}\n'
          f'targets_val shape\t{targets_val.shape}\n'
          f'feats_train_orig shape\t{feats_train_orig.shape}\n'
          f'targets_train_orig shape\t{targets_train_orig.shape}\n'
          f'feats_val_orig shape  \t{feats_val_orig.shape}\n'
          f'targets_val_orig shape\t{targets_val_orig.shape}\n')

    return {
        'data_train': data_train,
        'data_val': data_val,
        'scaler': scaler,
        'feats_train': feats_train,
        'targets_train': targets_train,
        'feats_val': feats_val,
        'targets_val': targets_val,
        'feats_train_orig': feats_train_orig,
        'targets_train_orig': targets_train_orig,
        'feats_val_orig': feats_val_orig,
        'targets_val_orig': targets_val_orig,
    }


def subsample_dataset(features_ds, targets_ds, features_ds_orig, targets_ds_orig, subsample_percent_size=1.0, debug=False):
    n = int(len(features_ds) * subsample_percent_size)
    sample_idxs = np.random.choice(np.arange(len(features_ds)), n, replace=False)
    x_sample = tf.constant(features_ds[sample_idxs])
    x_sample_orig = tf.constant(features_ds_orig[sample_idxs])
    y_sample = tf.constant(targets_ds[sample_idxs])
    y_sample_orig = tf.constant(targets_ds_orig[sample_idxs])
    if debug:
        print(f'x_sample type = {type(x_sample)}\tshape={x_sample.shape}\n'
              f'  Brunel_P     Brunel_ETA  nTracks_Brunel\n{x_sample[:3]}\n')
        print(f'y_sample type = {type(y_sample)}\tshape={y_sample.shape}\n'
              f'  RichDLLe     RichDLLk    RichDLLmu   RichDLLp    RichDLLbt\n{y_sample[:3]}')

    return x_sample, y_sample, x_sample_orig, y_sample_orig
