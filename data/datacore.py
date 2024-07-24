from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from torchvision import transforms
from dataclasses import dataclass
import torch
from sklearn.preprocessing import LabelEncoder
import data.data_transformation as my_transforms
import data.myconstants as constants

class NormalDataset(Dataset):
    def __init__(self, X, y, pids, ids, is_pseudo=False, transform=None):
        self.X = torch.Tensor(X).float()
        if not is_pseudo:
            self.y = torch.FloatTensor(y)
        else:
            self.y = np.zeros(len(X))
            self.y -= 1

            for idx, pseudo_label in y:
                assert self.y[idx] == -1
                self.y[idx] = pseudo_label
            
            self.y = torch.FloatTensor(self.y)
            assert (self.y != -1).all()

        self.pids = pids
        self.ids = ids
        self.transform = transform
        self.is_psuedo = is_pseudo

    def to_dataloader(self, cfg, shuffle=False):
        return torch.utils.data.DataLoader(
            self,
            batch_size=cfg.model_tta.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            x, y, idx, p
        """
        x = self.X[idx, :]
        y = self.y[idx]
        p = self.pids[idx]
        # id = self.ids[idx]

        if self.transform is not None:
            x = x.transpose(0, 1)
            x = self.transform(x.numpy())

        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = torch.FloatTensor(x[i].copy()).transpose(0, 1).float()

        return x, y, idx, p

    def __len__(self):
        return len(self.X)


def encode_labels(x):
    x_unique = np.unique(x)
    # Sort the unique labels
    x_unique = np.sort(x_unique)
    # Create a dictionary mapping the unique labels to the range [0, num_unique_labels]
    encoder_map = dict(zip(x_unique, range(len(x_unique))))
    # Encode the labels
    
    # Create a decoder array
    decoder_map = dict(zip(range(len(x_unique)), x_unique))
    return np.array([encoder_map[xi] for xi in x]), decoder_map

import logging

def create_ds_subset(ds, ratio_labeled):
    # For each class & person, select ratio_labeled indices
    selected_idxs = []

    for label in np.unique(ds.Y):
        label_idxs = np.where(ds.Y == label)[0]
        for person in np.unique(ds.P[label_idxs]): # For each person for which this class is present
            person_idxs = np.where(ds.P == person)[0]
            label_person_idxs = np.intersect1d(label_idxs, person_idxs)
            num_label_person_idxs = len(label_person_idxs)
            num_label_person_idxs_to_select = max(int(num_label_person_idxs * ratio_labeled), 1)
            selected_idxs.append(np.random.choice(label_person_idxs, num_label_person_idxs_to_select, replace=False))
    
    selected_idxs = np.concatenate(selected_idxs)
    logging.info("! >>>>>> Effective ratio labeled: {:.4f}".format(len(selected_idxs) / len(ds.Y)))
    unselected_idxs = np.setdiff1d(np.arange(len(ds.Y)), selected_idxs)
    return selected_idxs, unselected_idxs

@dataclass
class DSWrapper:
    X: any
    Y: any
    P: any
    ids: any
    name: str = None

    def copy(self):
        return DSWrapper(self.X, self.Y, self.P, self.ids, self.name)

    @staticmethod
    def with_encoded_PIDs_and_labels(name, X, Y, P, ids):
        Y = LabelEncoder().fit_transform(Y)
        wrapper = DSWrapper(X, Y, P, ids, name)
        return wrapper

    def to_normal_dataset(self, transform=None):
        return NormalDataset(self.X, self.Y, self.P, self.ids, transform=transform)

    def __len__(self):
        assert len(self.X) == len(self.Y) and len(self.Y) == len(self.P)
        return len(self.X)

    def __getitem__(self, idx):
        return DSWrapper(self.X[idx], self.Y[idx], self.P[idx], self.ids[idx], self.name)


@dataclass
class DSSplit:
    train: any
    val: any
    test: any

def SSL_generate_labels(x, cfg):
    label = [0, 0, 0, 0]

    if cfg.task.time_reversal:
        choice = np.random.choice(2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio])[0]
        x = my_transforms.flip(x, choice)
        label[constants.TIME_REVERSAL_POS] = choice

    if cfg.task.permutation:
        choice = np.random.choice(2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio])[0]
        x = my_transforms.permute(x, choice)
        label[constants.PERMUTATION_POS] = choice

    if cfg.task.time_warped:
        choice = np.random.choice(2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio])[0]
        x = my_transforms.time_warp(x, choice)
        label[constants.TIME_WARPED_POS] = choice

    x = torch.Tensor(np.array(x))
    label = torch.Tensor(np.array(label))
    return x, label

class SSL_dataset:
    def __init__(
        self,
        X,
        cfg,
        transform=None,
        shuffle=False,
    ):
        """
        Args:
            data_root (string): directory containing all data files
            file_list_path (string): file list
            cfg (dict): config
            shuffle (bool): whether permute epoches within one subject
            is_epoch_data (bool): whether each sample is one
            second of data or 10 seconds of data


        Returns:
            data : transformed sample
            labels (dict) : labels for avalaible transformations
        """
        self.cfg = cfg
        self.X = X
        self.shuffle = shuffle
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_data = self.X[idx]
        
        x_data, labels = SSL_generate_labels(x_data, self.cfg)
        x_data = np.asarray(x_data)

        if self.transform:
            x_data = np.transpose(x_data, (1, 0))
            x_data = self.transform(x_data)
            x_data = np.transpose(x_data, (1, 0))

        x_data = torch.FloatTensor(x_data)

        return (
            x_data,
            labels[constants.TIME_REVERSAL_POS],
            labels[constants.SCALE_POS],
            labels[constants.PERMUTATION_POS],
            labels[constants.TIME_WARPED_POS],
        )

# This function generates train-test splits for the input data based on the configuration settings in `cfg`.
# If `cfg.data.held_one_subject_out` is True, leave-one-subject-out splits are generated using the `LeaveOneGroupOut` class from scikit-learn.
# Otherwise, `cfg.num_split` number of train-test splits are generated using the `GroupShuffleSplit` class from scikit-learn, with a test size of 0.2 and a random state of 42.
# The function returns an iterator that generates the train-test splits.
def get_train_test_split(cfg, ds):
    if cfg.data.held_one_subject_out:
        print(f">>> TRAIN/TEST Split: Performing LeaveOneGroupOut.")
        folds = LeaveOneGroupOut().split(ds.X, ds.Y, groups=ds.P)
    else:
        print(f">>> TRAIN/TEST Split: Performing GroupShuffleSplit with {cfg.num_split} splits.")
        folds = GroupShuffleSplit(
            cfg.num_split, test_size=0.2, random_state=42
        ).split(ds.X, ds.Y, groups=ds.P)
    return folds

def temporal_resample(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """

    if X.shape[1] == length:
        print("No resampling needed")
        return X

    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )

    return X

def get_class_weights(y):
    import collections

    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)

    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)

    return weights

def get_data_with_subject_count(subject_count, ds):
    subject_list = np.unique(ds.P)

    if subject_count == len(subject_list):
        valid_subjects = subject_list
    else:
        valid_subjects = subject_list[:subject_count]

    pid_filter = [my_subject in valid_subjects for my_subject in ds.P]
    return ds[pid_filter]

def get_train_val_split(cfg, train_val_idxs, test_idxs, ds, cur_fold_idx) -> tuple:
    ds_train_val = ds[train_val_idxs]

    # assert intersection between train and test is empty
    assert len(np.intersect1d(train_val_idxs, test_idxs)) == 0

    # when we are not using all the subjects
    if cfg.data.subject_count != -1:
        print("Using only {} subjects".format(cfg.data.subject_count))
        ds_train_val = get_data_with_subject_count(cfg.data.subject_count, ds_train_val)

    # When changing the number of training data, we
    # will keep the test data fixed
    if cfg.data.held_one_subject_out:
        folds = LeaveOneGroupOut().split(
            ds_train_val.X, ds_train_val.Y, ds_train_val.P
        )
    else:
        val_size = 0.125 # 10% of train_val is val
        # We further divide up train into 70/10 train/val split
        folds = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=41
        ).split(ds_train_val.X, ds_train_val.Y, groups=ds_train_val.P)

    folds = list(folds)
    train_idxs, val_idxs = folds[(cur_fold_idx + 1) % len(folds)]

    assert len(np.intersect1d(train_idxs, val_idxs)) == 0
    return train_idxs, val_idxs

