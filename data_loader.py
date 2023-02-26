import sys
import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class RAMDataLoader:

    def __init__(self, chosen_dataset, data_dir) -> None:
        self.chosen_dataset = chosen_dataset
        self.data_dir = data_dir

        if self.chosen_dataset=="MNIST":
            # define transforms
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            trans = transforms.Compose([transforms.ToTensor(), normalize])

            # load dataset
            self.dataset_train = datasets.MNIST(self.data_dir, train=True, download=True, transform=trans)
            self.dataset_test = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
        elif self.chosen_dataset=="CIFAR10":
            # define transforms
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            trans = transforms.Compose([transforms.ToTensor(), normalize])

            # load dataset
            self.dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans)
            self.dataset_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans)
        else:
            print("This network is not thought for dataset ", chosen_dataset)
            sys.exit(-1)


    def get_train_valid_loader(
        self,
        batch_size,
        random_seed,
        valid_size=0.1,
        shuffle=True,
        show_sample=False,
        num_workers=4,
        pin_memory=False,
    ):
        """Train and validation data loaders.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Args:
            data_dir: path directory to the dataset.
            batch_size: how many samples per batch to load.
            random_seed: fix seed for reproducibility.
            valid_size: percentage split of the training set used for
                the validation set. Should be a float in the range [0, 1].
                In the paper, this number is set to 0.1.
            shuffle: whether to shuffle the train/validation indices.
            show_sample: plot 9x9 sample grid of the dataset.
            num_workers: number of subprocesses to use when loading the dataset.
            pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
                True if using GPU.
        """
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert (valid_size >= 0) and (valid_size <= 1), error_msg

        num_train = len(self.dataset_train)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # visualize some images
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=9,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy()
            X = np.transpose(X, [0, 2, 3, 1])
            plot_images(X, labels)

        return (train_loader, valid_loader)


    def get_test_loader(self, batch_size, num_workers=4, pin_memory=False):
        """Test datalaoder.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Args:
            data_dir: path directory to the dataset.
            batch_size: how many samples per batch to load.
            num_workers: number of subprocesses to use when loading the dataset.
            pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
                True if using GPU.
        """

        data_loader = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return data_loader

    def get_train_table_loader(self, batch_size, num_workers=4, pin_memory=False):
        """It's as `get_test_loader` but it actually takes training data.
        """

        data_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return data_loader
