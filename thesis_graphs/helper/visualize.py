from typing import Any
import torch
import umap
import plotly.express as px
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from skimage.future import graph

from torch.utils.data import DataLoader, Dataset
from thesis.helper import utils
from sklearn.preprocessing import StandardScaler

def visualize_segments(dataset, idx = 0):
  """
  Plots the boundaries of segments from a torch Dataset object.
  Input: dataset with the following attr: image, segments, idx img to plot
  Requirements: torch, skimage(img_as_float)
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  fig, ax = plt.subplots(1, figsize=(10, 10))
  plt.imshow(
      mark_boundaries(
          dataset[idx]["image"], 
          dataset[idx]["segments"]
          )
      )
  plt.show()

def visualize_rag_graph(dataset, idx = 0):
  """
  Plot rag mean-color graph on top of original picture
  Args:
    torch dataset-object containing "image" and "segments"
    idx of the graph to be displayed
  Requirements:
    matplotlib.pyplot as plt
    numpy as np
    torch
    skimage.future.graph
  """
  img = np.array(dataset[idx]["image"])
  segments = np.array(dataset[idx]["segments"])
  g = graph.rag_mean_color(img, segments)

  #Create plot
  fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

  #Create dummy mask of all ones for white background (bottom img)
  dummy = np.ones_like(img, dtype = np.float64)

  #Specify the fraction of the plot area that will be used to draw the colorbar
  #Set title and plot
  ax[0].set_title('RAG drawn with default settings')
  lc = graph.show_rag(segments, g, img, ax=ax[0], border_color ="white")
  fig.colorbar(lc, fraction=0.1, ax=ax[0])

  #Plot again on white background
  ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
  lc = graph.show_rag(segments, g, dummy,
                      img_cmap='gray', 
                      edge_cmap='viridis', 
                      ax=ax[1], border_color ="white")
  ax[1].imshow(dummy)
  fig.colorbar(lc, fraction=0.1, ax=ax[1])

  #Set axes off
  for a in ax:
      a.axis('off')

  plt.tight_layout()
  plt.show()

def visualize_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_neighbors: int = 30, 
    min_dist: int = 0,
    n_components: int = 3,
    prototypes: np.array = None,
    labels_dict: dict = {0.0:"fold", 1.0: "regular", 2.0: "gap", 3.0: "p_fold", 4.0: "p_reg", 5.0: "p_gap"},
    projected: bool = True,
    ) -> [list, list]:

    """
    Args: pretrained model; n_neighbors, min_dist, n_components for UMAP algo

    Returns: plot of embeddings in 3d-space
    """

    utils.set_parameter_requires_grad(model, False)

    #Calculate embeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)
    outs = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for data, data_2, label in dataloader:
        torch.no_grad()
        if projected == False:
            model.backbone_1.fc.register_forward_hook(get_activation("fc_1"))
            model.backbone_2.fc.register_forward_hook(get_activation("fc_2"))
            outs_proj = model(data.float(),data_2.float()).squeeze()

            a = activation["fc_1"]
            b = activation["fc_2"]
            features = torch.cat((a, b), 1)
            outs = torch.cat((outs, features), 0)
            labels = torch.cat((labels, label), 0)

        else:
            outs = torch.cat((outs, model(data.float(), data_2.float()).squeeze().to(device)), 0)
            outs = outs.detach().cpu()
            labels = torch.cat((labels, label), 0)

    feature_size = 0.5*outs.size()[1]
    outs = torch.cat((outs[:, :int(feature_size)], outs[:, int(feature_size):]), 0)
    labels = labels.detach().cpu().repeat(2).numpy()

    if prototypes is not None:
        labels_prototypes = [(np.max(labels) + 1 + i).astype(float) for i, proto in enumerate(prototypes)]
        labels = np.concatenate((labels, labels_prototypes))
        outs = torch.cat((outs, prototypes), 0)

    #Scale inputs to ease UMAP operation (m = 0, std = 1)
    scaler = StandardScaler()
    out_np = outs.detach().cpu().numpy()
    outs_scaled = scaler.fit_transform(out_np)

    #Generate px.scatter_3d plot
    sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

    clusterable_embedding = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        n_components = n_components,
        random_state = 47,
    ).fit_transform(out_np)

    df = pd.DataFrame()
    df["x"] = clusterable_embedding[:,0]
    df["y"] = clusterable_embedding[:,1]
    df["z"] = clusterable_embedding[:,2]
    df["labels"] = labels
    df["labels"] = df["labels"].replace(labels_dict)

    if prototypes is None:
        means = df.groupby("labels").mean()
        means["labels"] = ["p_fold", "p_regular", "p_gap"]
        df = df.append(means)

    fig = px.scatter_3d(df, x = "x", y = "y", z = "z", color = "labels")
    fig.show()
    if prototypes is None:
        return clusterable_embedding, labels, df.means.to_numpy()
    else:
        return clusterable_embedding, labels