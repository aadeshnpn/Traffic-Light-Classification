import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, datasets
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
from constants import EVAL_ID_MAP, SIMPLIFIED_CLASSES

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402

from PIL import Image

# See if the cuda is avaliable and store it in device
# assert torch.cuda.is_available(), "Change to gpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def load_pickle(fdir, name):
    import pathlib
    pathlib.Path(fdir).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(fdir, name)
    import pickle
    return pickle.load(open(fname, "rb"))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class BSTLDataset(Dataset):
  def __init__(
      self, root='/home/aadeshnpn/Documents/tlight/bsltd',
      size=512, train=True):

    fname = 'bsltd_train.pkl' if train else 'bsltd_test.pkl'
    self.data = load_pickle(root, fname)
    self.train = train
    print('datalenght', self.train, len(self.data))
    self.transform = get_transform(train)

  def __getitem__(self, index):
    image = Image.open(self.data[index]['path']).convert('RGB')
    target = self.data[index]['boxes']
    image = self.transform(image)
    # print(image.shape, target)
    boxes = []
    labels = []
    for t in target:
      boxes.append(
        [t['x_min'], t['y_min'], t['x_max'], t['y_max']])
      labels.append(
        [EVAL_ID_MAP[SIMPLIFIED_CLASSES[t['label']]]])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    labels = torch.reshape(labels, (labels.shape[0],))
    # print(boxes.shape, labels.shape)
    # if boxes.shape[0] ==1:
    #  print(boxes, labels)
    return image, {'boxes': boxes, 'labels': labels}

  def __len__(self):
    return 100 # len(self.data)


def run_one_epoch(
    train_loader, optimizer, model, lr_scheduler, device):

  model.train()
  losslog = []
  for images, targets in train_loader:
    image = list(image.to(device) for image in images)
    # print(targets[0])
    # targets = {k: v.to(device) for k, v in targets.items()}
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    # print(targets)
    loss_dict = model(image, targets)
    losses = sum(loss for loss in loss_dict.values())
    losslog.append(losses.item())

    # Reset the grad value to zero
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    if lr_scheduler is not None:
      lr_scheduler.step()
  return np.mean(losslog)


@torch.no_grad()
def evaluate(model, test_loader, device):
  model.eval()
  losslog = []
  for images, targets in test_loader:
    image = list(image.to(device) for image in images)
    # target = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

    # Predicted value
    outputs = model(image)
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    res = {target["labels"].item(): output for target, output in zip(targets, outputs)}
    losslog.append(res)
  return res


def train():
  import torchvision
  from torchvision.models.detection import FasterRCNN
  from torchvision.models.detection.rpn import AnchorGenerator
  # load a model pre-trained pre-trained on COCO
  from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True)
  num_classes = 4  # 1 class (person) + background
  # get number of input features for the classifier
  # in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  backbone = torchvision.models.mobilenet_v2(pretrained=True).features
  # FasterRCNN needs to know the number of
  # output channels in a backbone. For mobilenet_v2, it's 1280
  # so we need to add it here
  backbone.out_channels = 1280

  # let's make the RPN generate 5 x 3 anchors per spatial
  # location, with 5 different sizes and 3 different aspect
  # ratios. We have a Tuple[Tuple[int]] because each feature
  # map could potentially have different sizes and
  # aspect ratios
  anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

  # let's define what are the feature maps that we will
  # use to perform the region of interest cropping, as well as
  # the size of the crop after rescaling.
  # if your backbone returns a Tensor, featmap_names is expected to
  # be [0]. More generally, the backbone should return an
  # OrderedDict[Tensor], and in featmap_names you can choose which
  # feature maps to use.
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                  output_size=7,
                                                  sampling_ratio=2)

  # put the pieces together inside a FasterRCNN model
  model = FasterRCNN(backbone,
                    num_classes=4,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
  # replace the pre-trained head with a new one
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  # device = 'cpu'
  # use our dataset and defined transformations
  dataset = BSTLDataset()
  dataset_test = BSTLDataset(train=False)
  indices = torch.randperm(len(dataset)).tolist()
  dataset = torch.utils.data.Subset(dataset, indices[:-50])
  dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
  # define training and validation data loaders
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=2, shuffle=True, num_workers=4,
      collate_fn=collate_fn)

  data_loader_test = torch.utils.data.DataLoader(
      dataset_test, batch_size=1, shuffle=False, num_workers=4,
      collate_fn=collate_fn)

  # move model to the right device
  model.to(device)

  # construct an optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                              momentum=0.9, weight_decay=0.0005)

  # and a learning rate scheduler
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=3,
                                                  gamma=0.1)

  # let's train it for 10 epochs
  num_epochs = 2

  for epoch in tqdm(range(num_epochs)):
    # train for one epoch, printing every 10 iterations
    run_one_epoch(data_loader, optimizer, model, lr_scheduler, device)
    evaluate(model, data_loader_test, device=device)

  print("That's it!")


def main():
  train_dataset = BSTLDataset()
  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=5,
    shuffle=False, num_workers=1, pin_memory=True)
  for img, box in train_loader:
        print(img.shape)


# main()
train()