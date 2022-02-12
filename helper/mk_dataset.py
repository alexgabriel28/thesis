import os
import numpy as np
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch_geometric.utils import from_networkx
import networkx as nx

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage import measure
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dir_length(dir):
  initial_count = 0
  for path in os.listdir(dir):
      if os.path.isfile(os.path.join(dir, path)):
          initial_count += 1
  return initial_count

def collate_batch(batch):
  
  timage_list, graph_list, = [], []
  
  for _timage, _graph in batch:
    timage_list.append(_timage)
    graph_list.append(_graph)
  
  #timage_list = torch.tensor(timage_list, dtype=torch.int64)
  
  #graph_list = torch_geometric.data.Data(graph_list, batch_first=True, padding_value=0)
  
  return timage_list, graph_list

class InitDataset(Dataset):
  import torch
  """Face Landmarks dataset."""
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
      
      ##Assign edge weights: (color_gradient, manh_dist_1/25.5, manh_dist_2/25.5)
      manhattan = []
      agg_weights = []
      nx.set_edge_attributes(g, manhattan, "manhattan")
      nx.set_edge_attributes(g, manhattan, "agg_weights")
      for i, j in g.edges:
        g[i][j]["manhattan"] = (
            (abs(g.nodes[i]["centroid"][0] - g.nodes[j]["centroid"][0])), 
            abs(g.nodes[i]["centroid"][1] - g.nodes[j]["centroid"][1])
            )
        g[i][j]["agg_weights"] = (g[i][j]["weight"], g[i][j]["manhattan"][0], g[i][j]["manhattan"][1])

      #Convert networkx to torch_geometric graph
      gg = from_networkx(g)
      gg["mean color"][:,1] = gg["centroid"][:,0]/255
      gg["mean color"][:,2] = gg["centroid"][:,1]/255
      gg.x = gg["mean color"]
      gg.pos = gg["centroid"]
      gg.edge_attr = gg["agg_weights"]
      gg.y = class_id

      sample = {'image': np.array(image), 'tensor_image': tensor_image, "segments": segments, "graph": gg, "networkx": g, "class_id": class_id}
      return sample["tensor_image"], sample["graph"]
