import os
import torch
from dir_length import dir_length
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from torch_geometric.utils import from_networkx
from skimage.future import graph
from skimage import measure
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
      self.root_dir = root_dir
      self.transform = transform
      self.dir_list = os.listdir(self.root_dir)

  def __len__(self):
      return dir_length(self.root_dir)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_name = os.path.join(self.root_dir,
                              self.dir_list[idx])
      image = Image.open(img_name).convert("RGB")

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
        g[i][j]["agg_weights"] = (g[i][j]["weight"], g[i][j]["manhattan"][0]/25.5, g[i][j]["manhattan"][1]/25.5)

      #Convert networkx to torch_geometric graph
      gg = from_networkx(g)
      gg.x = g.nodes.data["mean color"]
      gg.edge_attr = g.edges.data["agg_weights"]
      sample = {'image': image, 'tensor_image': tensor_image, "segments": segments, "graph": gg, "networkx": g}
      return sample