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
    n_components: int = 3) -> [Any]:

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
    for data, data_2, label in dataloader:
        torch.no_grad()
        outs = torch.cat((outs, model(data.float(),data_2.float()).squeeze()), 0)
        labels = torch.cat((labels, label), 0)

    #Scale inputs to ease UMAP operation (m = 0, std = 1)
    scaler = StandardScaler()
    out_np = outs.detach().cpu().numpy()
    outs_scaled = scaler.fit_transform(out_np)

    #Generate px.scatter_3d plot
    sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

    clusterable_embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist = min_dist,
        n_components=n_components,
        random_state=47,
    ).fit_transform(out_np)

    df = pd.DataFrame()
    df["x"] = clusterable_embedding[:,0]
    df["y"] = clusterable_embedding[:,1]
    df["z"] = clusterable_embedding[:,2]

    fig = px.scatter_3d(df, x = "x", y = "y", z = "z", color = labels.cpu().numpy())
    fig.show()