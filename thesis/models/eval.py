import torch
from torch.utils.data import DataLoader, Dataset
from thesis.helper import utils

from typing import Any
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

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

def cal_linclf_acc(model: torch.nn.Module = None, 
                  train_dataloader: torch.utils.data.DataLoader = None,
                  test_dataloader: torch.utils.data.DataLoader = None,
                  clfs: dict = {
                       "KNeighbors": KNeighborsClassifier(),
                       "NearestCentroid": NearestCentroid(),
                       "SVC": SVC(gamma = "auto")
                       }, 
                  train_sz: float = 0.8, 
                  path: str = None
                  ) -> [Any]:

    utils.set_parameter_requires_grad(model, False)

    #Calculate embeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    outs = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    outs_test = torch.Tensor().to(device)
    labels_test = torch.Tensor().to(device)

    for data, data_2, label in train_dataloader:
        torch.no_grad()
        outs = torch.cat((outs, model(data.float(),data_2.float()).squeeze()), 0)
        labels = torch.cat((labels, label), 0)
    outs_cpu, labels_cpu = outs.detach().cpu().numpy(), labels.detach().cpu().numpy()

    for data, data_2, label in test_dataloader:
        torch.no_grad()
        outs_test = torch.cat((outs_test, model(data.float(),data_2.float()).squeeze()), 0)
        labels_test = torch.cat((labels_test, label), 0)
    outs_test_cpu = outs_test.detach().cpu().numpy()
    labels_test_cpu = labels_test.detach().cpu().numpy()

    # One Hot Encoder for ROC-AUC measure
    ohe = OneHotEncoder()
    ohe.fit(labels_cpu.reshape(-1, 1))

    with open(path, "a") as f:
        for k, v in clfs.items():
          pipeline = make_pipeline(StandardScaler(), v)
          pipeline.fit(outs_cpu, labels_cpu)
          predictions = pipeline.predict(outs_test_cpu)
          acc = accuracy_score(labels_test_cpu, predictions)

          lb= ohe.fit_transform(labels_test_cpu.reshape(-1, 1))
          pred = ohe.fit_transform(predictions.reshape(-1, 1))
          roc = roc_auc_score(lb.toarray(), pred.toarray())
          print(f"Clf: {k} | Acc: {acc} | AUC-ROC: {roc}", file = f)
          print(
              "------------------------------------------------------------------------------------", 
              file = f
              )