import os.path
import cv2
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from multiprocessing.pool import ThreadPool
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
class CMDataset(Dataset):
    def __init__(self, config, transforms = None, max_nums=None):
        super(CMDataset, self).__init__()
        self.config = config
        self.conditional = config.conditional
        self.image_size = config.image_size
        self.images_files_list, self.labels = config.images_file_list,  config.labels_list
        if (max_nums != None):
            self.images_files_list, self.labels = self.images_files_list[:max_nums], self.labels[:max_nums]
        self.ni = len(self.images_files_list)
        self.images_list = [None] * self.ni
        if transforms == None:
            self.transforms = T.Compose(
                        [
                            T.Resize((config.image_size, config.image_size)),
                            T.CenterCrop(config.image_size),
                            T.RandomHorizontalFlip(0.5),
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ])
        else:
            self.transforms = transforms
        self.get_images()

    def __len__(self):
        return len(self.images_files_list)
    def __getitem__(self, index):
        img = self.images_list[index]
        if self.conditional:
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
    def show_image(self, idx):
        img = Image.open(self.images_files_list[idx]).convert("RGB")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

class CIFAR10CMDataset(Dataset):
    def __init__(self, config, transforms = None, max_nums=None):
        super(CIFAR10CMDataset, self).__init__()
        self.conditional = config.conditional
        if transforms == None:
            self.transforms = T.Compose(
                [
                    T.Resize((config.image_size, config.image_size)),
                    T.CenterCrop(config.image_size),
                    T.RandomHorizontalFlip(0.5),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ])
        else:
            self.transforms = transforms
        self.train_images_labels = [(img, label)for img, label in CIFAR10(".", True, download=True)]
        self.valid_images_labels = [(img, label) for img, label in CIFAR10(".", False, download=True)]
        self.images_labels = self.train_images_labels + self.valid_images_labels
        if(max_nums!=None):
            self.images_labels = self.images_labels[:max_nums]
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