
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='.', batch_size=512, n_workers=20):
        super().__init__()
        self.data_dir = data_dir
        self.bs = batch_size
        self.nw = n_workers
        self.train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        self.test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )


    def setup(self, stage: str):
        # download first, and change download to false in compute canada
        self.train = CIFAR10(root=self.data_dir, train=True, download=True, transform=self.train_transforms)
        self.val, self.test = random_split(CIFAR10(root=self.data_dir, train=False, transform=self.test_transforms), [5000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.bs, num_workers=self.nw)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.bs, num_workers=self.nw)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.bs, num_workers=self.nw)