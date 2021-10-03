from torchvision import transforms
from image_toolbox.dataset_utils.tranform_utils import FaceAlignTransform

def get_pre_transforms(input_shape):
    """get transforms before applying image augmentation"""
    return transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[2])),
    ])

def get_transforms(resize_shape, mode='train'):
    """ get finishing transforms"""
    if mode == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    elif mode == 'val' or mode == 'test':
        return transforms.Compose([
            get_pre_transforms(resize_shape),
            transforms.ToTensor(),
        ])
    elif mode == 'inference':
        return transforms.Compose([
            get_pre_transforms(resize_shape),
            FaceAlignTransform(shape=resize_shape[1]),
            transforms.ToTensor(),
        ])