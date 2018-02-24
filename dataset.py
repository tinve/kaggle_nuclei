import numpy as np
import torch
import cv2

from torch.utils.data import Dataset


def load_image(sample_path):
    sample_name = str(sample_path).split('/')[-1]
    img = cv2.imread(str(sample_path/sample_name) + '.png')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(sample_path):
    sample_name = str(sample_path).split('/')[-1]
    mask = np.load(str(sample_path/sample_name) + '.npy')
    return (mask > 0).astype(np.uint8)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


class NucleiDataset(Dataset):
    """
    __getitem__ method returns image and its mask for training set;
                image and path to its directory for testing set
    """
    def __init__(self, sample_paths, set_type='train', transform=None):
        self.sample_paths = sample_paths
        self.set_type = set_type
        self.transform = transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]

        img = load_image(sample_path)

        if self.set_type == 'train':
            mask = load_mask(sample_path)

            assert mask.shape[0] == img.shape[0]
            assert mask.shape[1] == img.shape[1]

            img, mask = self.transform(img, mask)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()

        img = self.transform(img)
        return to_float_tensor(img), sample_path
