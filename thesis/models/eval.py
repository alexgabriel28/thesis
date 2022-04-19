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
    for data, data_2, label in dataloader:
        torch.no_grad()
        outs = torch.cat((outs, model(data.float(),data_2.float()).squeeze()), 0)
        labels = torch.cat((labels, label), 0)
    return outs, labels

def eval_few_shot(model, dataloader, dataloader_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proto = []
    dists = []
    model.eval()
    model.to(device)
    outs = torch.Tensor().to(device)
    outs_train = torch.Tensor().to(device)

    labels = torch.Tensor().to(device)
    labels_train = torch.Tensor().to(device)

    for data, data_2, label in dataloader:
        torch.no_grad()
        outs_train = torch.cat((outs_train, model(data.float(), data_2.float()).squeeze()), 0)
        labels_train = torch.cat((labels_train, label), 0)

    labels_train = labels_train.detach().cpu().repeat(2).numpy()
    outs_train = outs_train.detach().cpu()
    feature_size = 0.5*outs_train.size()[1]
    outs_train = torch.cat((outs_train[:, :int(feature_size)], outs_train[:, int(feature_size):]), 0).detach().cpu().numpy() 
    
    for label in np.unique(labels_train):
          proto.append(np.mean(outs_train[labels_train == label], 0))

    for data, data_2, label in dataloader_test:
        torch.no_grad()
        outs = torch.cat((outs, model(data.float(),data_2.float()).squeeze()), 0)
        labels = torch.cat((labels, label), 0)
    
    labels = labels.detach().cpu().repeat(2).numpy()
    outs = outs.detach().cpu()
    feature_size = 0.5*outs.size()[1]
    outs = torch.cat((outs[:, :int(feature_size)], outs[:, int(feature_size):]), 0)

    for instance in outs:
        distances = np.sum((proto - instance.detach().cpu().numpy())**2, 1)
        min_dist = np.argmin(distances)
        dists.append(min_dist)
    return min_dist, distances, proto