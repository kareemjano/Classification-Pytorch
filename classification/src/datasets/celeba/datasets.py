import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path

class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.map is not None:
            x = self.map(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

    def nb_classes(self):
        return self.dataset.nb_classes()


class ClassificationDataset(Dataset):
    """ Dataset with items being (image, label)"""

    def __init__(self, data_map, url="", transform=None, map_to_int=False,
                 offset_y=0, class_to_idx=None):
        """

        :param data_map: Dict containing labels as keys and an array of paths of the images corrisponding to each label.
        :param transform: transforms to be added.
        :param map_to_int: Maps the labels to int values. Used if the initial labels are strings.
        :param offset_y: Adds an offset to the int labels. Used if the labels start from 1 and not from 0.
        """
        self.offset_y = offset_y
        self.image_map = data_map
        self.class_to_idx = class_to_idx
        self.map_to_int = map_to_int
        self.url = url
        if map_to_int and self.class_to_idx is None:
            self.class_to_idx = dict()
            self.encode_classes()
        self.ys, self.im_paths = self._idx_people_encode()
        self.transform = transform

    def encode_classes(self):
        """
        encodes str labels to int
        :return:
        """
        for label in list(self.image_map.keys()):
            self.class_to_idx[label] = self.class_to_idx.get(label, len(self.class_to_idx))

    def _idx_people_encode(self):
        """Private function used for the index encoding of the dataset"""
        ys, im_paths = [], []
        for key, value in list(self.image_map.items()):
            for img_path in self.image_map[key]:
                label_id = self.class_to_idx[key] if self.map_to_int else key
                offset_id = label_id - self.offset_y

                ys.append(offset_id)
                im_paths.append(os.path.join(str(key), img_path))

        return ys, im_paths

    def set_transform(self, transform):
        """Set the transform attribute for image transformation"""
        self.transform = transform

    def nb_classes(self):
        return len(np.unique(self.ys))

    def __getitem__(self, idx):
        im = Image.open(Path(self.url) / self.im_paths[idx])
        if self.transform is not None:
            im = self.transform(im)

        return im, self.ys[idx]

    def __len__(self):
        return len(self.ys)