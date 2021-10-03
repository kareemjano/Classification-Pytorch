import torch
from torchvision import transforms
import numpy as np

from .datasets import MapDataset
from image_toolbox.dataset_utils.augmentation_utils import getImageAug
from image_toolbox.dataset_utils.tranform_utils import ToNumpy

def augment_dataset(dataset, transforms, p):
    """
    adds transformed samples to dataset
    :param dataset: dataset to augment
    :param transforms: transforms used to augment
    :param p: fraction of the dataset to augment [0, 1]
    :return: input dataset plus augmented dataset
    """
    num_augmented = int(len(dataset) * p)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    aug_dataset = torch.utils.data.Subset(dataset, indices[:num_augmented])
    aug_dataset = MapDataset(aug_dataset, transforms)
    return torch.utils.data.ConcatDataset([aug_dataset, dataset])

def get_augmentations():
    """get image augmentation transforms"""
    return transforms.Compose([
                ToNumpy(),
                getImageAug().augment_image,
            ])