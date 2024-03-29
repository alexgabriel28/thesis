from typing import Any
import torch
import umap
import plotly.express as px
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from thesis.helper import utils
from sklearn.preprocessing import StandardScaler

def visualize_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_neighbors: int = 30, 
    min_dist: int = 0,
    n_components: int = 3,
    prototypes: np.array = None,
    labels_dict: dict = {
        0.0: "regular", 
        1.0:"fold", 
        2.0: "gap", 
        3.0: "hole", 
        4.0: "rabbet",
        5.0: "p_regular",
        6.0: "p_fold",
        7.0: "p_gap",
        8.0: "p_hole",
        9.0: "p_rabbet"
        },
    projected: bool = True,
    outs_storage: torch.Tensor = None,
    label_storage: torch.Tensor = None,
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
            model.fc.register_forward_hook(get_activation("fc_1"))
            outs_proj = model(torch.cat((data.float(), data_2.float())))

            a = activation["fc_1"]

            outs = torch.cat((outs, a), 0)
            labels = torch.cat((labels, label.repeat(2)), 0)

        else:
            outs_1 = model(data.float())
            outs_2 = model(data_2.float())
            outs_cat = torch.cat((outs_1, outs_2)).to(device)
            outs = torch.cat((outs, outs_cat)).to(device)
            labels = torch.cat((labels, label.repeat(2)), 0)

    outs = torch.cat((outs, outs_storage))
    labels = torch.cat((labels, label_storage))
    #feature_size = 0.5*outs.size()[1]
    #outs = torch.cat((outs[:, :int(feature_size)], outs[:, int(feature_size):]), 0)
    labels = labels.detach().cpu().numpy()

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
        means["labels"] = ["p_fold", "p_gap", "p_hole", "p_rabbet", "p_regular"]
        df = df.append(means)

    fig = px.scatter_3d(df, x = "x", y = "y", z = "z", color = "labels")
    fig.show()
    if prototypes is None:
        return clusterable_embedding, labels
    else:
        return clusterable_embedding, labels

device = "cuda" if torch.cuda.is_available() else "cpu"