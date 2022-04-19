import os
import torch
import sys
import glob
import torch
from typing import Any

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def set_parameter_requires_grad(model, require_grad = True):
    if require_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def summary(model, f = None):
    params = list(model.named_parameters())
    line_sep = "-------------------------------------------------------------------------------------------------"
    print(line_sep, file = f)
    print("{:>30}  {:>30} {:>30}".format("Layer", "Shape", "No. Parameters"), file = f)
    print(line_sep, file = f)
    for elem in params:
      layer = elem[0]
      shape = list(elem[1].size())
      count = torch.tensor(elem[1].size()).prod().item()
      print("{:>30}  {:>30} {:>30}".format(layer, str(shape), str(count)), file = f)
    print(line_sep, file = f)
    sum_params = sum([param.nelement() for param in model.parameters()])
    print("Total Parameters:", sum_params, file = f)
    train_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
    print("Trainable Parameters:", train_params, file = f)
    print("Non-Trainable Parameters:", sum_params - train_params, file = f)
    print(line_sep, file = f)

def save_train_specs(model: torch.nn.Module, 
                     train_dict: dict, 
                     callbacks: list, 
                     rdir: str, 
                     file_name: str, 
                     comment: str = None,
                     save_model: bool = False) -> Any:

    file_dir = os.path.join(rdir, file_name)
    with open(file_dir, "a") as f:
        print("Model Summary:\n", file = f)
        print(model, file = f)
        print("-------------------------------------------------------------------------", file = f)
        print("Parameter Summary: \n", file = f)
        summary(model, f = f)
        print("-------------------------------------------------------------------------", file = f)
        print("Train Specifications:", file = f)
        f.write(repr(train_dict) + "\n")
        f.write("-------------------------------------------------------------------------")
        if comment != None:
          f.write(comment)
          f.write("\n-------------------------------------------------------------------------")
    callback_dir = rdir + file_name.split(".")[0] + "_callbacks.pt"
    torch.save(callbacks, callback_dir)
    if save_model == True:
      model_dir = rdir + file_name.split(".")[0] + "_model.pt"
      torch.save(model, model_dir)

def cal_linclf_acc(model: torch.nn.Module = None, 
                   dataloader: torch.utils.data.DataLoader = None, 
                   clfs: dict = {
                       "KNeighbors": KNeighborsClassifier(),
                       "NearestCentroid": NearestCentroid(),
                       "SVC": SVC(gamma = "auto")
                       }, 
                   train_sz: float = 0.8, 
                   path: str = None
                   ) -> [Any]:

    set_parameter_requires_grad(model, False)

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
    outs_cpu, labels_cpu = outs.detach().cpu().numpy(), labels.detach().cpu().numpy()

    # Calculate linear classifier accuracies
    data_train, data_test, labels_train, labels_test = train_test_split(
        outs_cpu, labels_cpu, train_size = train_sz
        )
    
    # One Hot Encoder for ROC-AUC measure
    ohe = OneHotEncoder()
    ohe.fit(labels_cpu.reshape(-1, 1))

    with open(path, "a") as f:
        for k, v in clfs.items():
          pipeline = make_pipeline(StandardScaler(), v)
          pipeline.fit(data_train, labels_train)
          predictions = pipeline.predict(data_test)
          acc = accuracy_score(labels_test, predictions)

          lb= ohe.fit_transform(labels_test.reshape(-1, 1))
          pred = ohe.fit_transform(predictions.reshape(-1, 1))
          roc = roc_auc_score(lb.toarray(), pred.toarray())
          print(f"Clf: {k} | Acc: {acc} | AUC-ROC: {roc}", file = f)
          print(
              "------------------------------------------------------------------------------------", 
              file = f
              )