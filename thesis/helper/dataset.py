import os
import numpy as np
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import networkx as nx

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float, img_as_int
from skimage.future import graph
from skimage.color import gray2rgb
from skimage import measure
from PIL import Image

from sklearn.model_selection import train_test_split

import augly.image as imaugs
import augly

from tqdm.auto import tqdm

from thesis.helper import tensor_img_transforms

#Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Count length of directory
def dir_length(dir):
  initial_count = 0
  for path in os.listdir(dir):
      if os.path.isfile(os.path.join(dir, path)):
          initial_count += 1
  return initial_count

################################################################################
#######################        Augmentation    #################################
################################################################################
COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 0.4,
    "saturation_factor": 0.2,
    "p": 0.8
}

AUGMENTATIONS = [
    imaugs.Resize(224, 224),            
    #imaugs.Blur(),
    #imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    #imaugs.transforms.HFlip(p=0.5),
    #imaugs.RandomNoise(mean = 0.0, var = 0.1, seed = 42, p = 0.2),
    #imaugs.transforms.Contrast(factor = 1.7),
    imaugs.Brightness(1.5),
  ]

TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
TENSOR_TRANSFORMS = transforms.Compose(
    AUGMENTATIONS + [transforms.ToTensor(), 
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                     )

################################################################################
#Collate function -> return image data for ResNet backbone testing #############
################################################################################
def collate_model_batch(batch):
  image_list, label_list = [], []
  # Transform in eval model: Only normalization (resizing happened before)
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
  ])
  for image, graph in batch:
    label_list.append(graph.y)
    image_list.append(transform(image))
  
  elem = image_list[0]
  numel = sum(x.numel() for x in image_list)
  storage = elem.storage()._new_shared(numel)
  out = elem.new(storage).resize_(len(batch), *list(elem.size()))

  return torch.stack(image_list, 0, out=out).squeeze().to(
      device, non_blocking = True
      ), torch.tensor(label_list).to(device)

################################################################################
#Collate function -> for self-supervised learning; transforms image (Tensor)####
################################################################################
def collate_ss_batch(batch):
  timage_list, graph_list = [], []
  for _timage, _graph in batch:
    graph = Data(
        x = _graph.x, 
        edge_attr = _graph.edge_attr, 
        edge_index = _graph.edge_index,
        y = _graph.y
        )
    t = tensor_img_transforms.Transform()
    y1, y2 = t(_timage)
    timage_list.extend([y1, y2])
    graph_list.extend([graph, graph])

  elem = timage_list[0]
  numel = sum(x.numel() for x in timage_list)
  storage = elem.storage()._new_shared(numel)
  out = elem.new(storage).resize_(len(batch), *list(elem.size()))

  return torch.stack(timage_list, 0, out=out).squeeze().to(
      device, non_blocking = True
      ), Batch.from_data_list(graph_list).to(device, non_blocking = True)

################################################################################
#Collate function -> for self-supervised learning; transforms images (Tensor)###
################################################################################
def collate_CNN_2(batch):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  timage_list_1, timage_list_2, label_list = [], [], []
  for timage, labels in batch:
    t = tensor_img_transforms.Transform()

    y1, y2 = t(timage)
    y3, y4 = t(timage)

    timage_list_1.extend([y1, y3])
    timage_list_2.extend([y2, y4])
    label_list.extend([labels for i in range(2)])

  elem_1 = timage_list_1[0]
  numel_1 = sum(x.numel() for x in timage_list_1)
  storage_1 = elem_1.storage()._new_shared(numel_1)
  out_1 = elem_1.new(storage_1).resize_(len(batch), *list(elem_1.size()))

  elem_2 = timage_list_2[0]
  numel_2 = sum(x.numel() for x in timage_list_2)
  storage_2 = elem_2.storage()._new_shared(numel_2)
  out_2 = elem_2.new(storage_2).resize_(len(batch), *list(elem_2.size()))

  elem_3 = label_list[0]
  numel_3 = sum(x.numel() for x in label_list)
  storage_3 = elem_3.storage()._new_shared(numel_3)
  out_3 = elem_3.new(storage_3).resize_(len(batch), *list(elem_3.size()))

  return torch.stack(timage_list_1, 0, out=out_1).squeeze().to(
      device, non_blocking = True
      ), torch.stack(timage_list_2, 0, out=out_2).squeeze().to(
      device, non_blocking = True
      ), torch.stack(label_list, 0, out=out_3).squeeze().to(
      device, non_blocking = True
      )

################################################################################
#ImageDataset -> takes image_dir ###############################################
################################################################################
class ImageDataset(Dataset):
  def __init__(self, root_dir: str, classes: list):
    self.root_dir = root_dir
    self.classes = classes
    self.label_list = []
    self.image_list = []
    self.class_map = {}
    cls_list = []
    count = 0
    for class_path in glob.glob(root_dir + "*"):
        cls = class_path.split("/")[-1]
        cls_list.append(cls)
        if count in self.classes:
            for img_path in glob.glob(class_path + "/*.png"):
              img = Image.open(str(img_path)).convert("RGB")
              tensor_image = TF.pil_to_tensor(img)
              self.image_list.append(tensor_image)
              self.label_list.append(cls)
        count += 1
    
    for i, cls in enumerate(cls_list):
        self.class_map[cls] = i

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    class_name = self.label_list[idx]
    if torch.is_tensor(idx):
          idx = idx.tolist()

    self.class_id = self.class_map[class_name]
    self.class_id = torch.tensor([self.class_id])
    return self.image_list[idx], self.class_id

################################################################################
#ImageDataset for continual learning -> takes image_dir ########################
################################################################################

class ContImageDataset(Dataset):
  """
  Dataset class to read images from root_dir, where images of one class are stored
  in specific folders for continual learning setting. 
  Creates Dataset to provide to torch.utils.data.Dataloader
  """
  def __init__(self, root_dir: str, classes: list):
    self.root_dir = root_dir
    self.classes = classes
    self.label_list = []
    self.image_list = []
    self.class_map = {}
    cls_list = []
    count = 0
    for class_path in glob.glob(root_dir + "*"):
        cls = class_path.split("/")[-1]
        cls_list.append(cls)
        if count in self.classes:
            for img_path in glob.glob(class_path + "/*.png"):
              img = Image.open(str(img_path)).convert("RGB")
              tensor_image = TF.pil_to_tensor(img)
              self.image_list.append(tensor_image)
              self.label_list.append(cls)
        count += 1
    
    for i, cls in enumerate(cls_list):
        self.class_map[cls] = i

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    class_name = self.label_list[idx]
    if torch.is_tensor(idx):
          idx = idx.tolist()

    self.class_id = self.class_map[class_name]
    self.class_id = torch.tensor([self.class_id])
    return self.image_list[idx], self.class_id

################################################################################
# ImageDatasetTrain - ImageDatasetTest #########################################
################################################################################
class ImageDatasetTest(Dataset):
  def __init__(self, root_dir: str, classes: list, value = 0.8):
    self.root_dir = root_dir
    self.classes = classes
    self.label_list = []
    self.image_list = []
    self.class_map = {}
    self.idx_dict = {}
    cls_list = []
    count = 0

    for class_path in glob.glob(root_dir + "*"):
        cls = class_path.split("/")[-1]
        cls_list.append(cls)
        self.idx_dict[cls] = []
        if count in self.classes:
            for i, img_path in enumerate(glob.glob(class_path + "/*.png")):
              img = Image.open(str(img_path)).convert("RGB")
              tensor_image = TF.pil_to_tensor(img)
              if np.random.rand() > value:
                  self.image_list.append(tensor_image)
                  self.label_list.append(cls)
                  self.idx_dict[cls].append(i)
        count += 1
    
    for i, cls in enumerate(cls_list):
        self.class_map[cls] = i

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    class_name = self.label_list[idx]
    if torch.is_tensor(idx):
          idx = idx.tolist()

    self.class_id = self.class_map[class_name]
    self.class_id = torch.tensor([self.class_id])
    return self.image_list[idx], self.class_id
  
  def get_idx(self):
    return self.idx_dict

class ImageDatasetTrain(Dataset):
  def __init__(self, root_dir: str, classes: list, idx_dict):
    self.root_dir = root_dir
    self.classes = classes
    self.label_list = []
    self.image_list = []
    self.class_map = {}
    cls_list = []
    count = 0
    for class_path in glob.glob(root_dir + "*"):
        cls = class_path.split("/")[-1]
        cls_list.append(cls)
        idx_list = idx_dict[cls]

        if count in self.classes:
            for i, img_path in enumerate(glob.glob(class_path + "/*.png")):
              if i not in idx_list:
                  img = Image.open(str(img_path)).convert("RGB")
                  tensor_image = TF.pil_to_tensor(img)
                  self.image_list.append(tensor_image)
                  self.label_list.append(cls)
        count += 1
    
    for i, cls in enumerate(cls_list):
        self.class_map[cls] = i

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    class_name = self.label_list[idx]
    if torch.is_tensor(idx):
          idx = idx.tolist()

    self.class_id = self.class_map[class_name]
    self.class_id = torch.tensor([self.class_id])
    return self.image_list[idx], self.class_id

################################################################################
#ImageDataset for continual learning -> takes image_dir ########################
################################################################################
def create_dataloader_cont(
    num_classes: str = 5,
    cls_per_run: int = 2,
    batch_size: int = 128, 
    root_dir: str = "/content/drive/MyDrive/MT Gabriel/data_ext/"
    ) -> torch.utils.data.DataLoader:
    """
    This function returns two lists: dataloader_list and dataloader_test_list,
    both containing dataloaders for a continual learning setting, i.e., the 
    classes are provided sequentially, according to the specified Args.
    Args:
        num_classes: str: (Default_value = 5) Number of total classes
        cls_per_run: int: (Default value = 2) Number of classes per dataloader
        batch_size: int: (Default value = 128) Batch size
        root_dir: str: Directory of images: each image class must be in a separate
            folder in root_dir
    Returns:
        (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
        dataloader_list for training, dataloader_list for testing
    """
    dataloader_list = []
    dataloader_test_list = []
    cls_list = []
    cls_list_per_run = []

    for i in range(num_classes - 1):
        cls_list_per_run = []
        for j in range(cls_per_run):
            ####################################################################
            ### Change 0 to i to receive sequential cls_list, i.e. [0,1,2,3,...]
            cls_list_per_run.extend([i + j])
        cls_list.append(cls_list_per_run)

    for i, classes in enumerate(cls_list):
        image_dir = root_dir
        image_dataset = ContImageDataset(image_dir, classes = classes)

        labels = [label.numpy() for tensor, label in iter(image_dataset)]
        train_indices, test_indices = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels)
        train_dataset = torch.utils.data.Subset(image_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(image_dataset, test_indices)

        dataloader = DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            shuffle = True, 
            pin_memory = False,
            collate_fn = collate_CNN_2
            )
        dataloader_list.append(dataloader)

        dataloader_test = DataLoader(
            test_dataset, 
            batch_size = batch_size,
            shuffle = True,
            pin_memory = False,
            collate_fn = collate_CNN_2
            )
        dataloader_test_list.append(dataloader_test)
    return dataloader_list, dataloader_test_list

#Train-test-split: 80:20
def train_test_split(dataset, test_ratio = 0.2, stratify = True):
    """
    Helper function to split dataset
    """
    if stratify == False:
        train_size = int((1-test_ratio) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
          dataset, [train_size, test_size]
          )
    else:
        labels = [label for tensor, label in iter(dataset)]
        train_indices, test_indices = train_test_split(list(range(len(labels))), test_size=test_ratio, stratify=labels)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset

################################################################################
# Create Dataloaders ###########################################################
################################################################################
def create_dataloader(
    num_classes: str = 5,
    cls_per_run: int = 2,
    batch_size: int = 128,
    drop_last: bool = False,
    train_size: float = 0.8,
    root_dir: str = "/content/drive/MyDrive/MT Gabriel/data_ext/"
    ) -> torch.utils.data.DataLoader:

    dataloader_list = []
    dataloader_test_list = []
    cls_list_train = []
    cls_list_test = []
    idx_dict_list = []

    for i in range(num_classes - (cls_per_run - 1)):
        cls_list_per_run = []
        cls_list_test_per_run = []
        for j in range(cls_per_run):
            ####################################################################
            ### Change 0 to i to receive sequential cls_list, i.e. [0,1,2,3,...]
            cls_list_per_run.extend([i + j])
            cls_list_test_per_run.extend(list(range(i+j+1)))
        cls_list_test.append(cls_list_test_per_run)

    for i, classes in enumerate(cls_list_test):
        image_dir = root_dir
        image_dataset = ImageDatasetTest(image_dir, classes = classes, value = train_size)
        idx_dict_list.append(image_dataset.get_idx())

        dataloader_test = DataLoader(
            image_dataset, 
            batch_size = batch_size,
            shuffle = True,
            pin_memory = False,
            collate_fn = collate_CNN_2
            )
        dataloader_test_list.append(dataloader_test)

    for i in range(num_classes - (cls_per_run - 1)):
        cls_list_per_run = []
        for j in range(cls_per_run):
            ####################################################################
            ### Change 0 to i to receive sequential cls_list, i.e. [0,1,2,3,...]
            cls_list_per_run.extend([i + j])
        cls_list_train.append(cls_list_per_run)

    for i, classes in enumerate(cls_list_train):
        image_dir = root_dir
        image_dataset = ImageDatasetTrain(image_dir, classes = classes, idx_dict = idx_dict_list[i])

        dataloader = DataLoader(
            image_dataset, 
            batch_size = batch_size, 
            shuffle = True, 
            pin_memory = False,
            collate_fn = collate_CNN_2,
            drop_last = drop_last,
            )
        dataloader_list.append(dataloader)

    return dataloader_list, dataloader_test_list

# Make data with different failures
def make_failure_data(
    bg_folder_path, 
    overlay_path, 
    output_path,
    x_lim = [0.3, 0.8],
    y_lim = [0.3, 0.8],
    overlay = [0.1, 0.5],
    ):
    for img_path in glob.glob(bg_folder_path + "/*.png"):
        overlay_size = float(np.random.choice(np.arange(
            overlay[0]*10, overlay[1]*10
            ), 1)/10)
        x_pos = float(np.random.choice(np.arange(x_lim[0]*10, x_lim[1]*10), 1)/10)
        y_pos = float(np.random.choice(np.arange(y_lim[0]*10, y_lim[1]*10), 1)/10)

        background_img = Image.open(str(img_path)).convert("RGB")
        output_img = output_path + img_path.split("/")[-1]

        augly.image.functional.overlay_image(
            overlay = overlay_path, 
            image = background_img, 
            output_path=output_img, 
            opacity=1.0, 
            overlay_size=overlay_size, 
            x_pos=x_pos, 
            y_pos=y_pos,
            )