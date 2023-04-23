import os
import cv2
import torch
import json
from PIL import Image
from tqdm.auto import tqdm
import pytorch_lightning as pl
from argparse import Namespace
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from typing import List, Optional, Tuple, Dict, Union

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
def get_data_list(data_path:str,
                  modes: Optional[List]=None) -> Tuple:
    labels, images = [], []
    if not modes:
        modes = ['train', 'valid', 'test']
    for mode in modes:
        file_path = os.path.join(data_path, f'{mode}.txt')
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                images.append(os.path.join(data_path, line[0]))
                labels.append(int(line[1]))
    return images, labels
class CMDataset(Dataset):
    def __init__(self,
                 config:Namespace) -> None:
        super(CMDataset, self).__init__()
        self.config = config
        self.images_files_list, self.labels = get_data_list(config.dataset_path, config.modes)
        if (config.max_nums != None):
            self.images_files_list, self.labels = self.images_files_list[:config.max_nums], self.labels[:config.max_nums]
        self.ni = len(self.images_files_list)
        self.images_list = [None] * self.ni
        self.transforms = T.Compose(
            [
                T.Resize((self.config.image_size, self.config.image_size)),
                T.CenterCrop(self.config.image_size),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])
        self.get_images()
        self.index2label = self.get_index2label(config.dataset_path) if config.conditional else None
    def __len__(self):
        return len(self.images_files_list)
    def __getitem__(self,
                    index:int) -> Union[torch.Tensor, Tuple]:
        img = self.images_list[index]
        if self.config.conditional:
            return self.transforms(img), torch.tensor(self.labels[index])
        else:
            return self.transforms(img)
    def get_images(self):
        gb = 0
        fcn = cv2.imread
        # fcn = self.imread
        results = ThreadPool(NUM_THREADS).imap(fcn, self.images_files_list)
        pbar = tqdm(enumerate(results), total=self.ni, mininterval=0.3)
        for i, x in pbar:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            gb += x.nbytes
            x = Image.fromarray(x)
            self.images_list[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
            pbar.desc = f"train Caching images ({gb / 1E9:.1f}GB)"
        pbar.close()
    def get_index2label(self,
                        dataset_path:str) -> Dict:
        with open(os.path.join(dataset_path, 'index2label.json'), "r", encoding="utf-8") as f:
            index2index = json.load(f)
            label2index = {value: int(key) for key, value in index2index.items()}
            index2label = {value: key for key, value in label2index.items()}
        return index2label
    def show_image(self,
                   idx:int) -> None:
        img = Image.open(self.images_files_list[idx]).convert("RGB")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

class CIFAR10CMDataset(Dataset):
    def __init__(self,
                 config:Namespace) -> None:
        super(CIFAR10CMDataset, self).__init__()
        self.conditional = config.conditional
        self.transforms = T.Compose(
            [
                T.Resize((config.image_size, config.image_size)),
                T.CenterCrop(config.image_size),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])

        self.train_images_labels = [(img, label)for img, label in CIFAR10(".", True, download=True)]
        self.valid_images_labels = [(img, label) for img, label in CIFAR10(".", False, download=True)]
        self.images_labels = self.train_images_labels + self.valid_images_labels
        if(config.max_nums!=None):
            self.images_labels = self.images_labels[:config.max_nums]
        cifar10_classes = CIFAR10('.', False, download=False).classes
        self.index2label = {i:cls for i, cls in enumerate(cifar10_classes)} if config.conditional else None
    def __len__(self):
        return len(self.images_labels)
    def __getitem__(self, idx):
        if self.conditional:
            return self.transforms(self.images_labels[idx][0]), self.images_labels[idx][1]
        else:
            return self.transforms(self.images_labels[idx][0])
    def show_image(self, idx):
        plt.imshow(self.images_labels[idx][0])
        plt.axis("off")
        plt.show()
def get_dataset(config:Namespace) -> Dataset:
    return eval(config.dataset_name)(config)
class CMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(CMDataModule, self).__init__()
        self.config = config
    def setup(self, stage: str) -> None:
        self.train_dataset =get_dataset(self.config)
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=self.config.workers,
                          pin_memory=True,
                          drop_last=True)
def get_datamodule(config:Namespace) -> pl.LightningDataModule:
    return CMDataModule(config)