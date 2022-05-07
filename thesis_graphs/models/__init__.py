##Init-file for model
import os
import torch
from torch import nn
import wandb

from typing import Any, Dict, List, Sequence
import torchvision
from torchvision.models import resnet50
from torchsummary import summary
import torch_geometric
from tqdm.auto import tqdm
from thesis.models import GraphClassificationModel as gcm
from thesis.models import GraphClassificationPNA as pna