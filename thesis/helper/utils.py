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
    """
    Set requires_grad of all parameters in the provided model.
    Args:
        model: torch.nn.Module
        require_grad: bool: (Default value = True)
    Return:
        model: torch.nn.Module: Model with param.requires_grad as specified
    """
    if require_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def summary(model):
  """
  Creates model summary
  Args
      model: torch.nn.Module: a torch model
  Returns
      [Any] -> .txt file with a summary of the parameters of the model,
          with specification of trainable and non-trainable parameters
  """
  params = list(model.named_parameters())
  line_sep = "-------------------------------------------------------------------------------------------------"
  print(line_sep)
  print("{:>30}  {:>30} {:>30}".format("Layer", "Shape", "No. Parameters"))
  print(line_sep)
  for elem in params:
    layer = elem[0]
    shape = list(elem[1].size())
    count = torch.tensor(elem[1].size()).prod().item()
    print("{:>30}  {:>30} {:>30}".format(layer, str(shape), str(count)))
  print(line_sep)
  sum_params = sum([param.nelement() for param in model.parameters()])
  print("Total Parameters:", sum_params)
  train_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
  print("Trainable Parameters:", train_params)
  print("Non-Trainable Parameters:", sum_params - train_params)
  print(line_sep)

def save_train_specs(model: torch.nn.Module, 
                     train_dict: dict, 
                     callbacks: list, 
                     rdir: str, 
                     file_name: str, 
                     comment: str = None,
                     save_model: bool = False) -> Any:
    """
    Writes the model train specifications to the specified file_name
    """
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