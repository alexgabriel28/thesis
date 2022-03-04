import os
import torch
import sys
import glob
import torch
from typing import Any

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
                     dir: str, 
                     file_name: str, 
                     comment: str = None) -> Any:

    file_dir = os.path.join(dir, file_name)
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
    callback_dir = dir + file_name.split(".")[0] + "_callbacks" + ".pt"
    torch.save(callbacks, callback_dir)
