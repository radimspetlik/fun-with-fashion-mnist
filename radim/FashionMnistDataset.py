from torch.utils.data import Dataset
import numpy as np


class NoAugmentation(object):
    def augment_image(self, image):
        return image

    def to_deterministic(self):
        pass


class FashionMnistDataset(Dataset):
    '''Generates data'''

    def __init__(self, config, X, y, is_trn=True):
        'Initialization'
        self.config = config
        self.dim = (config['model']['input_size_h'], config['model']['input_size_w'])
        self.augment = config['trn']['augment']

        self.X = X
        self.y = y.astype(np.long)

        self.n_channels = config['model']['input_channels']
        self.is_trn = is_trn

        self.X = self.X.reshape((-1, *self.dim))

        if self.augment and self.is_trn:
            self.cnn_augmentation = self.prepare_augmentation_pipeline(self.config)
        else:
            self.cnn_augmentation = NoAugmentation()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Generate data
        X = self.cnn_augmentation.augment_image(self.X[index])

        return X[np.newaxis].astype(np.float32), self.y[index]

    def prepare_augmentation_pipeline(self, config):
        from imgaug.parameters import DiscreteUniform
        br_change = config['trn']['brightness_augmentation_amount']  # (was 70)
        contrast_min = config['trn']['augmentation_contrast_min']
        contrast_max = config['trn']['augmentation_contrast_max']
        rotation = config['trn']['augmentation_rotation']
        if rotation == 'uniform':
            rotation = DiscreteUniform(0, 360)
        translation = config['trn']['augmentation_translation']
        scale_min = config['trn']['augmentation_scale_min']
        scale_max = config['trn']['augmentation_scale_max']
        shear = config['trn']['augmentation_shear']

        from imgaug import augmenters as iaa
        return iaa.Sequential(
            [
                iaa.Add((-br_change, br_change), per_channel=False),  # change brightness of images (by -10 to 10 of original value)
                iaa.contrast.LinearContrast((contrast_min, contrast_max), per_channel=False),  # improve or worsen the contrast
                iaa.Affine(rotate=rotation,
                           translate_percent={"x": (-translation, translation), 'y': (-translation, translation)},
                           scale=(scale_min, scale_max),
                           shear=(-shear, shear),
                           cval=128),
                iaa.Fliplr(p=0.5)
            ],
            random_order=True
        )
