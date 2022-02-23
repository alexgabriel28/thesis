import os
import numpy as np
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch_geometric.transforms as T
import torchvision.transforms.functional as TF
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch, Data
import networkx as nx

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float, img_as_int
from skimage.future import graph
from skimage.color import gray2rgb
from skimage import measure
from PIL import Image

import augly.image as imaugs

from tqdm.auto import tqdm

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
#Collate function -> return graph and image data for simultaneous processing####
################################################################################

def collate_batch(batch):
  timage_list, graph_list = [], []
  for _timage, _graph in batch:
    graph = Data(
        x = _graph.x, 
        edge_attr = _graph.edge_attr, 
        edge_index = _graph.edge_index
        )
    
    timage_list.append(_timage)
    graph_list.append(graph)

  elem = timage_list[0]
  numel = sum(x.numel() for x in timage_list)
  storage = elem.storage()._new_shared(numel)
  out = elem.new(storage).resize_(len(batch), *list(elem.size()))

  return torch.stack(timage_list, 0, out=out).squeeze().to(
      device, non_blocking = True
      ), Batch.from_data_list(graph_list).to(device, non_blocking = True)

################################################################################
#Collate function -> return graph data for GNN testing #########################
################################################################################
def collate_graph_batch(batch):
  graph_list = []
  for _timage, _graph in batch:
    graph = Data(
        x = _graph.x, 
        edge_attr = _graph.edge_attr, 
        edge_index = _graph.edge_index,
        y = _graph.y
        )
    graph_list.append(graph)
  return Batch.from_data_list(graph_list).to(device, non_blocking = True)

################################################################################
#Create dataset with internal processing of images and graph creation###########
#Note: very slow __getitem__ -> use create_datalist funct & LigthDataset class##
################################################################################

class InitDataset(Dataset):
  import torch
  """Create dataset from raw images with transformation and graph creation"""
  def __init__(self, root_dir, transform=None):
      """
      Args:
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied 
          on a sample.

      Requirements:
          torch
          torch.transforms.functional as TF
          torch_geometric.utils.from_networkx
          numpy as np
          skimage.future.graph
          skimage.segmentation.slic
          skimage.util.img_as_float
          skimage.measure
          networkx
          dir_length (helper submodule)
      """
      self.imgs_path = root_dir
      file_list = glob.glob(self.imgs_path + "*")
      print(file_list)
      self.data = []
      self.img_path = []
      self.class_name = []
      for class_path in file_list:
          class_name = class_path.split("/")[-1]
          for img_path in glob.glob(class_path + "/*.png"):
              self.data.append([img_path, class_name])
              self.img_path.append(img_path)
              self.class_name.append(class_name)
      print(self.data)
      print(self.img_path)
      print(self.class_name)

      self.class_map = {"fold" : 0, "regular": 1, "gap": 2}
      self.root_dir = root_dir
      self.transform = transform
      self.dir_list = os.listdir(self.root_dir)

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      img_path = self.img_path[idx]
      class_name = self.class_name[idx]
      if torch.is_tensor(idx):
          idx = idx.tolist()

      image = Image.open(str(img_path)).convert("RGB")

      class_id = self.class_map[class_name]
      class_id = torch.tensor([class_id])

      if self.transform:
          image = self.transform(image)

      segments = slic(
          img_as_float(image),
          n_segments=500, 
          compactness=30, 
          sigma=0.3, 
          start_label=1
          )
      
      tensor_image = TF.pil_to_tensor(image)
      g = graph.rag_mean_color(np.array(img_as_float(image)), np.array(segments))

      #Assign centroid coordinates for the graph
      regions = measure.regionprops(segments)
      centroid = []
      nx.set_node_attributes(g, centroid, "centroid")

      for (n, data), region in zip(g.nodes(data=True), regions):
        g.nodes[n]["centroid"] = tuple(map(int, region['centroid']))
      
      ##Assign edge weights: (color_gradient, manh_dist_1/255, manh_dist_2/255)
      manhattan = []
      agg_weights = []
      nx.set_edge_attributes(g, manhattan, "manhattan")
      nx.set_edge_attributes(g, manhattan, "agg_weights")
      for i, j in g.edges:
        g[i][j]["manhattan"] = (
            (abs(g.nodes[i]["centroid"][0] - g.nodes[j]["centroid"][0])), 
            abs(g.nodes[i]["centroid"][1] - g.nodes[j]["centroid"][1])
            )
        g[i][j]["agg_weights"] = (
            g[i][j]["weight"], 
            g[i][j]["manhattan"][0], 
            g[i][j]["manhattan"][1]
            )

      #Convert networkx to torch_geometric graph
      gg = from_networkx(g)
      gg["mean color"][:,1] = gg["centroid"][:,0]/255
      gg["mean color"][:,2] = gg["centroid"][:,1]/255
      gg.x = gg["mean color"]
      gg.pos = gg["centroid"]
      gg.edge_attr = gg["agg_weights"]
      gg.y = class_id

      sample = {
          'image': np.array(image), 
          'tensor_image': tensor_image, 
          "segments": segments, 
          "graph": gg, 
          "networkx": g, 
          "class_id": class_id
          }
      return sample["tensor_image"], sample["graph"]

################################################################################
#Create timage_list/ graph_list from img dir for processing w/ LightDataset#####
################################################################################
def create_datalists(
    root_dir: str = "/content/drive/MyDrive/MT Gabriel/data_1/", 
    graph_dir: str = None,
    image_dir: str = None,
    transforms = TRANSFORMS,
    n_segments = 500,
    )->[list, list]:

  imgs_path = root_dir
  file_list = glob.glob(imgs_path + "*")

  data = []
  img_paths = []
  class_names = []

  
  graph_list = []
  image_list = []

  #Class mapping
  class_map = {"fold" : 0, "regular": 1, "gap": 2}
  dir_list = os.listdir(root_dir)
  for class_path in file_list:
      class_name = class_path.split("/")[-1]
      for img_path in glob.glob(class_path + "/*.png"):
          data.append([img_path, class_name])
          img_paths.append(img_path)
          class_names.append(class_name)

  #Loop over data instance
  for idx, img_path in tqdm(enumerate(img_paths)):
    class_name = class_names[idx]
    if torch.is_tensor(idx):
      idx = idx.tolist()

    #Open image as PIL Image
    image = Image.open(str(img_path)).convert("RGB")

    #Map data instace to label
    class_id = class_map[class_name]
    class_id = torch.tensor([class_id])

    #Apply transform
    image = transforms(image)

    #Create segments using slic
    segments = slic(
        img_as_float(image),
        n_segments=n_segments, 
        compactness=30, 
        sigma=0.3, 
        start_label=1
        )
    
    #skimage.measure.regionprops only takes segments with labels > 0 into account
    #for some rare cases, slic produces a one-pixel segment with value 0 in the
    #top left corner -> set to 1 for intended performance
    if 0 in segments:
      segments[segments == 0] = 1

    #Convert PIL Image to torch.Tensor
    tensor_image = TF.pil_to_tensor(image)
    g = graph.rag_mean_color(np.array(img_as_float(image)), np.array(segments))

    #Assign centroid coordinates for the graph
    regions = measure.regionprops(segments)
    centroid = []
    nx.set_node_attributes(g, centroid, "centroid")

    #Calculate segment centroids
    for (n, data), region in zip(g.nodes.data(), regions):
      g.nodes[n]["centroid"] = tuple(map(int, region['centroid']))

    ##Assign edge weights: (color_gradient, manh_dist_1/255, manh_dist_2/255)
    manhattan = []
    agg_weights = []
    nx.set_edge_attributes(g, manhattan, "manhattan")
    nx.set_edge_attributes(g, manhattan, "agg_weights")

    #Calculate Manhattan distance between segment centroids and assign to edge
    #weights to preserve geometric information
    for i, j in g.edges:
      g[i][j]["manhattan"] = (
          abs(g.nodes[i]["centroid"][0] - g.nodes[j]["centroid"][0])/255, 
          abs(g.nodes[i]["centroid"][1] - g.nodes[j]["centroid"][1])/255
          )
      
      g[i][j]["agg_weights"] = (
          g[i][j]["weight"], 
          g[i][j]["manhattan"][0], 
          g[i][j]["manhattan"][1]
          )

    #Convert networkx to torch_geometric graph
    gg = from_networkx(g)
    gg["mean color"][:,1] = gg["centroid"][:,0]/255
    gg["mean color"][:,2] = gg["centroid"][:,1]/255
    gg.x = gg["mean color"]
    gg.pos = gg["centroid"]
    gg.edge_attr = gg["agg_weights"]
    gg.y = class_id

    #Store the augmented tensor_images and graphs in respective list
    graph_list.append(gg)
    image_list.append(tensor_image)

  if graph_dir is not None:
    torch.save(graph_list, graph_dir)
  if image_dir is not None:
    torch.save(image_list, image_dir)

  return graph_list, image_list

################################################################################
#LightDataset -> takes graph_list and image_list ###############################
################################################################################
class LightDataset(Dataset):
  import torch
  def __init__(self, graph_list, image_list):
    self.graph_list = graph_list
    self.image_list = image_list

  def __len__(self):
    return len(self.graph_list)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
          idx = idx.tolist()
    return self.image_list[idx], self.graph_list[idx]

from torch_geometric.utils import degree

#Train-test-split: 80:20
def train_test_split(dataset, test_ratio = 0.2):
    train_size = int((1-test_ratio) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset