import numpy as np
import pandas as pd
import os
from os.path import splitext
from typing import Union, List
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetTemplate(data.Dataset):
    """A dataset template class that supports reading and transforming images
    and segmentation labels. Labels are pre-stored as numpy array with data
    type np.int8. Subclasses should implement how to get self.img_names.
    """
    def __init__(self, img_dir, label_dir, transform):
        """
        :param img_dir: the directory where the images are stored
        :param label_dir: the directory where the labels are stored
        :param transform: the albumentations transformation applied to image and
        label
        """
        self.img_dir, self.label_dir = img_dir, label_dir
        self.img_names = []
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self._get_image(img_name)
        label = self._get_label(img_name)
        img, label = self._transform(img, label)
        return img, label, img_name

    def __len__(self):
        return len(self.img_names)

    def _get_image(self, img_name):
        img_path = f'{self.img_dir}/{img_name}'
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img

    def _get_label(self, img_name):
        base = img_name.rsplit('.', 1)[0]
        label_dir = f'{self.label_dir}/{base}.npy'
        label = np.load(label_dir).astype(np.int8)
        return label

    def _transform(self, img, label):
        img = np.array(img)
        transformed = self.transform(image=img, mask=label)
        img = transformed['image']
        label = transformed['mask']
        return img, label


class CSVSplitDataset(DatasetTemplate):
    """A dataset class that reads a csv split file containing (name, split)
    pairs to get the dataset consisting of images with or without the specified
    split number(s).
    """
    def __init__(self,
                 img_dir: str,
                 label_dir: str,
                 split_csv: str,
                 split_num: Union[int, List[int]],
                 transform,
                 split_col_name: str = "split",
                 reverse: bool = False):
        """
        :param split_csv: the path to the csv file that contains the split info
        :param split_num: the split number(s) of the image
        :param split_col_name: the name of the column in the csv file that
        contains the split number, defaults to "split"
        :param reverse: if True, the images without the split_num are selected,
        defaults to False
        """
        super().__init__(img_dir, label_dir, transform)
        if isinstance(split_num, (int, np.int64)):
            split_num = [split_num]
        df = pd.read_csv(split_csv)
        if reverse:
            self.img_names = list(df['name'][~df[split_col_name].isin(split_num)])
        else:
            self.img_names = list(df['name'][df[split_col_name].isin(split_num)])


class TextSplitDataset(DatasetTemplate):
    """A dataset class that reads a text split file containing the name of the
    images in the target dataset split.
    """
    def __init__(self, img_dir, label_dir, split_txt, transform):
        """
        :param split_txt: the path of the text file that contains the names of
        the images in the split
        """
        super().__init__(img_dir, label_dir, transform)
        self.img_names = np.loadtxt(split_txt, dtype=str, delimiter='\n', ndmin=1)


class FolderDataset(DatasetTemplate):
    """A dataset class that reads images from a folder. Only images with suffix
    .tif, .jpg, .png are taken. If labels are not provided, the output label is
    -1 everywhere.
    """
    def __init__(self, img_dir, label_dir, transform):
        super().__init__(img_dir, label_dir, transform)
        self.img_names = [name for name in os.listdir(self.img_dir)
                          if splitext(name)[1] in ['.tif', '.jpg', '.png']]
        self.no_label = label_dir is None

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self._get_image(img_name)
        if self.no_label:
            label = -np.ones_like(img)[:, :, 0]
        else:
            label = self._get_label(img_name)
        img, label = self._transform(img, label)
        return img, label, img_name


def get_list_of_ops(args, library):
    if args is None: return []
    ops = []
    for func_name in args:
        func = getattr(library, func_name)
        ops.append(func(**args[func_name]))
    return ops


def get_transform(args, is_train):
    """
    It takes in the arguments and a boolean value, and returns a transform
    object.
    :param args: the global arguments
    :param is_train: if the transform is for training or evaluating
    :return: transform operations to be performed on the image
    """
    if is_train:
        transform = A.Compose([
            A.RandomCrop(*args.train_size),
            *get_list_of_ops(args.augmentations, A),
            A.Normalize(mean=args.mean, std=args.std),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(*args.eval_size),
            A.Normalize(mean=args.mean, std=args.std),
            ToTensorV2()
        ])
    return transform


def visualize_augmentations(dataset, idx=0, n_samples=5):
    """
    It takes a dataset, an index, and a number of samples, and plots the
    augmented images and their masks for the image at the index.
    :param dataset: the dataset to visualize
    :param idx: the index of the image to visualize, defaults to 0
    :param n_samples: number of augmented samples to show, defaults to 5
    """
    import copy
    from matplotlib import pyplot as plt
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
    ])
    plt.figure(dpi=150)
    figure, axes = plt.subplots(nrows=n_samples, ncols=2, figsize=(6, 3*n_samples))
    for i in range(n_samples):
        image, mask, name = dataset[idx]
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, interpolation="nearest")
        axes[i, 0].set_title("Augmented image")
        axes[i, 1].set_title("Augmented mask")
        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


def get_dataloaders(args):
    transform_train = get_transform(args, is_train=True)
    transform_eval = get_transform(args, is_train=False)
    s_info = args.split_info
    if s_info.type == "CSVSplit":
        split_file_path = f"{args.dataset_root}/splits/{s_info.split_file}"
        if s_info.train_reverse:
            train_split_num = [s_info.val_split_num, s_info.test_split_num]
        else:
            train_split_num = s_info.train_split_num
        train_set = CSVSplitDataset(args.img_dir, args.label_dir,
                                    split_csv=split_file_path,
                                    split_num=train_split_num,
                                    transform=transform_train,
                                    split_col_name=s_info.split_col_name,
                                    reverse=s_info.train_reverse)
        val_set = CSVSplitDataset(args.img_dir, args.label_dir,
                                  split_csv=split_file_path,
                                  split_num=s_info.val_split_num,
                                  transform=transform_eval,
                                  split_col_name=s_info.split_col_name)
    elif s_info.type == "TextSplit":
        train_split_file_path = f"{args.dataset_root}/splits/{s_info.train_split_file}"
        val_split_file_path = f"{args.dataset_root}/splits/{s_info.val_split_file}"
        train_set = TextSplitDataset(args.img_dir, args.label_dir,
                                     split_txt=train_split_file_path,
                                     transform=transform_train)
        val_set = TextSplitDataset(args.img_dir, args.label_dir,
                                   split_txt=val_split_file_path,
                                   transform=transform_eval)
    else:
        raise NotImplementedError(args.split_info.type)
    if args.train_repeat > 1:
        train_set = data.ConcatDataset([train_set] * args.train_repeat)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # get test dataloader
    if s_info.test_type == "validation":
        test_set = val_set
    elif s_info.test_type == "TextSplit":
        split_file_path = f"{args.dataset_root}/splits/{s_info.test_split_file}"
        test_set = TextSplitDataset(args.img_dir, args.label_dir, split_file_path, transform_eval)
    elif s_info.test_type == "CSVSplit":
        split_file_path = f"{args.dataset_root}/splits/{s_info.split_file}"
        test_set = CSVSplitDataset(args.img_dir, args.label_dir, split_file_path,
                                   s_info.test_split_num, transform_eval, s_info.split_col_name)
    elif s_info.test_type == "folder":
        test_set = FolderDataset(s_info.test_img_dir, s_info.test_label_dir, transform_eval)
    else:
        raise NotImplementedError(s_info.test_type)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    from args import Arguments
    import sys
    sys.argv.extend(['--config', 'test_aug.yaml'])
    parser = Arguments()
    args = parser.parse_args(use_random_seed=False)
    args.train_repeat = 1
    train_loader, val_loader, test_loader = get_dataloaders(args)
    visualize_augmentations(train_loader.dataset, idx=0, n_samples=5)
