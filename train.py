import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import numpy as np
from tqdm import tqdm
import matplotlib


# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402

from PIL import Image

# See if the cuda is avaliable and store it in device
# assert torch.cuda.is_available(), "Change to gpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pickle(fdir, name):
    import pathlib
    pathlib.Path(fdir).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(fdir, name)
    import pickle
    return pickle.load(open(fname, "rb"))


class BSTLDataset(Dataset):
  def __init__(
      self, root='/home/aadeshnpn/Documents/tlight/bsltd',
      size=512, train=True):

    fname = 'bsltd_train.pkl' if train else 'bsltd_test.pkl'
    self.data = load_pickle(root, fname)
    self.transform = transforms.Compose(
         [transforms.ToTensor()])

  def __getitem__(self, index):
    image = Image.open(self.data[index]['path'])
    boxes = self.data[index]['boxes']
    image = self.transform(image)
    print(image.shape, boxes)
    return image, []

  def __len__(self):
    return  10    # len(self.data)


def main():
  train_dataset = BSTLDataset()
  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=5,
    shuffle=False, num_workers=1, pin_memory=True)
  for img, box in train_loader:
        print(img.shape, box)


main()