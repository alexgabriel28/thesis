import torch
from torch.utils.data import DataLoader, Dataset
from thesis.helper import utils

def model_eval(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ) -> [torch.Tensor, torch.Tensor]:

    """
    Args: pretrained model; dataloader: (returns image data, labels)
    Requires: thesis.helper.utils, torch
    Returns: plot of embeddings in 3d-space
    """
    utils.set_parameter_requires_grad(model, False)

    #Calculate embeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    outs = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    for data, label in dataloader:
        torch.no_grad()
        outs = torch.cat((outs, dataloader(data.float()).squeeze()), 0)
        labels = torch.cat((labels, label), 0)
    return outs, labels