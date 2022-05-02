from typing import Any, Dict, List, Sequence
from IPython.display import clear_output

import torch
import torch.nn as nn

import torchvision
from torchvision.models import resnet50
from torchvision.models import 

import torch_geometric

from tqdm.auto import tqdm

from thesis.models import GraphClassificationModel as gcm
from thesis.models import GraphClassificationPNA as pna
from thesis.loss import vicreg_loss_fn as vlf

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

class VICRegGraphGAT(nn.Module):
    def __init__(
        self,
        features_dim: int = 512,
        proj_output_dim: int = 4096,
        proj_hidden_dim: int = 4096,
        sim_loss_weight: float = 25,
        var_loss_weight: float = 25,
        cov_loss_weight: float = 1,
        sz_out: int = 64,
        backbone_1 = torchvision.models.resnet18(pretrained = True, progress = True, zero_init_residual=True),
        backbone_2 = gcm.GraphClassificationModel(torch_geometric.nn.GATv2Conv),
        **kwargs
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)
        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__()

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.sz_out = sz_out
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2

        self.backbone_1.fc = nn.Linear(in_features=features_dim, out_features = features_dim)
        
        # projector
        self.projector_1 = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        layers = []
        while self.sz_out < proj_output_dim:
          layers.append(nn.Linear(self.sz_out, 2*self.sz_out))
          layers.append(nn.BatchNorm1d(2*self.sz_out))
          layers.append(nn.ReLU())
          self.sz_out = 2*self.sz_out

        layers.append(nn.Linear(proj_output_dim, proj_output_dim))
        
        self.projector_2 = nn.ModuleList(layers)

    def forward(self, timage: torch.Tensor, graph: torch_geometric.data.Data, *args, **kwargs):
        """Performs the forward pass of the backbone and the projector.
        Args:
            X (torch.Tensor): a batch of images in the tensor format.
        """
        z1 = self.projector_1(self.backbone_1(timage))
        z2 = self.backbone_2(graph.x.float(), graph.edge_index, graph.edge_attr.float(), graph.batch).float().squeeze()

        for layer in self.projector_2:
            z2 = layer(z2)

        out = torch.cat((z1, z2), 1)
        return out


class VICRegGraphPNA(nn.Module):
    def __init__(
        self,
        features_dim: int = 512,
        proj_output_dim: int = 4096,
        proj_hidden_dim: int = 4096,
        sim_loss_weight: float = 25,
        var_loss_weight: float = 25,
        cov_loss_weight: float = 1,
        sz_out: int = 64,
        backbone_1 = torchvision.models.resnet18(pretrained = True, progress = True, zero_init_residual=True),
        backbone_2 = None,
        **kwargs
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)
        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__()

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.sz_out = sz_out
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2

        self.backbone_1.fc = nn.Linear(in_features=features_dim, out_features = features_dim)
        
        # projector
        self.projector_1 = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        layers = []
        while self.sz_out < proj_output_dim:
          layers.append(nn.Linear(self.sz_out, 2*self.sz_out))
          layers.append(nn.BatchNorm1d(2*self.sz_out))
          layers.append(nn.ReLU())
          self.sz_out = 2*self.sz_out

        layers.append(nn.Linear(proj_output_dim, proj_output_dim))
        
        self.projector_2 = nn.ModuleList(layers)

    def forward(self, timage: torch.Tensor, graph: torch_geometric.data.Data, *args, **kwargs):
        """Performs the forward pass of the backbone and the projector.
        Args:
            X (torch.Tensor): a batch of images in the tensor format.
        """
        z1 = self.projector_1(self.backbone_1(timage))
        z2 = self.backbone_2(graph.x.float(), graph.edge_index, graph.batch).float().squeeze()

        for layer in self.projector_2:
            z2 = layer(z2)

        out = torch.cat((z1, z2), 1)
        return out

class VICRegCNN_2(nn.Module):
    def __init__(
        self,
        features_dim: int = 512,
        proj_output_dim: int = 4096,
        proj_hidden_dim: int = 4096,
        sim_loss_weight: float = 25,
        var_loss_weight: float = 25,
        cov_loss_weight: float = 1,
        backbone_1 = torchvision.models.resnet18(pretrained = True, progress = True, zero_init_residual=True),
        backbone_2 = torchvision.models.resnet18(pretrained = True, progress = True, zero_init_residual=True),
        **kwargs
    ):

        """Implements VICReg with two CNN branches (https://arxiv.org/abs/2105.04906)
        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
            backbone_1, backbone_2 (torch.nn.Module): models of the respective branches
        """

        super().__init__()

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2

        self.backbone_1.fc = nn.Linear(in_features = features_dim, out_features = features_dim)
        self.backbone_2.fc = nn.Linear(in_features = features_dim, out_features = features_dim)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(
        self, 
        timage_1: torch.Tensor, 
        timage_2: torch.Tensor, 
        *args, 
        **kwargs):
      
        """Performs the forward pass of the backbones and the projectors.
        Args:
            X (torch.Tensor): a batch of images in the tensor format.
        """
        z1 = self.projector(self.backbone_1(timage_1))
        z2 = self.projector(self.backbone_2(timage_2))

        out = torch.cat((z1, z2), 1)
        return out