import pytorch_lightning as pl
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from image_toolbox.dataset_utils.celeba_utils import read_identity_file
from image_toolbox.dataset_utils.general_utils import list_to_map
from .tranforms import get_transforms, get_pre_transforms
from .augmentations import get_augmentations, augment_dataset
from .datasets import ClassificationDataset, MapDataset

import numpy as np

class Classification_Dataloader(pl.LightningDataModule):
    """Dataloader used to load the data for the training the liveness detection and the group loss models which use
    CelebA or LFW, and CFW respectively."""
    def __init__(self, conf, batch_size=32,
                 num_workers=4, image_aug_p=0):
        """
        :param name: Choice from DATASETS
        :param batch_size: batch size of the dataloader
        :param num_workers: number of workers to be used to load the data.
        :param input_shape: (Tuple) (channels, width, height).
        :param image_aug_p: if >0 image_aug_p*len(dataset) images will be augmented and added.
        """
        super().__init__()
        import os
        print("dataloader cwd", os.getcwd())
        self.base_url = Path(conf.datasets.celeba.base_url)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_aug_p = image_aug_p
        self.dataset_conf = conf.datasets.celeba
        self.model_select = conf.nets.select
        self.input_shape = tuple(conf.nets[self.model_select].params.input_shape)
        self.val_r, self.test_r = tuple(conf.datasets.celeba["val_test_r"])
        torch.manual_seed(0)

    def setup(self):
        transforms = get_transforms(self.input_shape, mode='train')
        pre_transforms = get_pre_transforms(self.input_shape)
        augmentations = get_augmentations()
        val_transforms = get_transforms(self.input_shape, mode='val')

        # split_labels_map = get_split_labels_map(label_txt_path=Path(self.dataset_conf.base_url) / self.dataset_conf.labels_txt_file,
        #                                         split_txt_path=Path(self.dataset_conf.base_url) / self.dataset_conf.split_txt_file)

        img_names, labels = read_identity_file(label_txt_path=(self.base_url / self.dataset_conf.labels_txt_file),
                                               min_samples_per_class=self.dataset_conf.min_samples_per_class)

        if self.dataset_conf.nb_samples > 0:
            sorted_indecies = np.argsort(labels)
            img_names = img_names[sorted_indecies][:self.dataset_conf.nb_samples]
            labels = labels[sorted_indecies][:self.dataset_conf.nb_samples]

        images_url = self.base_url / self.dataset_conf.zip_file_name.split(".")[0]

        val_test_r = self.val_r + self.test_r

        X_train, X_val_test, y_train, y_val_test = train_test_split(
        img_names, labels, test_size=val_test_r, shuffle=True, stratify=labels, random_state=45)

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=self.test_r/val_test_r, shuffle=True, stratify=y_val_test, random_state=45)

        self.train_dataset = ClassificationDataset(list_to_map(X_train, y_train), url=images_url, map_to_int=True)
        self.val_dataset = ClassificationDataset(list_to_map(X_val, y_val), url=images_url, map_to_int=True,
                                                 class_to_idx=self.train_dataset.class_to_idx)
        self.test_dataset = ClassificationDataset(list_to_map(X_test, y_test), url=images_url, map_to_int=True,
                                                  class_to_idx=self.train_dataset.class_to_idx)

        self.train_dataset = MapDataset(self.train_dataset, pre_transforms)
        # load the augmented dataset for a precentage of samples:
        if self.image_aug_p > 0:
            self.train_dataset = augment_dataset(self.train_dataset, augmentations, self.image_aug_p)
            print('size of augmented train dataset', len(self.train_dataset))

        self.train_dataset = MapDataset(self.train_dataset, transforms)
        self.val_dataset = MapDataset(self.val_dataset, val_transforms)
        self.test_dataset = MapDataset(self.test_dataset, val_transforms)
        print("size of train, val, test datasets",
              len(self.train_dataset),
              len(self.val_dataset),
              len(self.test_dataset))

        print("number of classes in train, val, test datasets",
              self.train_dataset.nb_classes(),
              self.val_dataset.nb_classes(),
              self.test_dataset.nb_classes())

    # return the dataloader for each split
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size * 2,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=None
                                           )

    def nb_classes(self):
        return self.train_dataset.nb_classes()

