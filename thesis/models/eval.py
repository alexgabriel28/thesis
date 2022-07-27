import torch
from torch.utils.data import DataLoader, Dataset
from thesis.helper import utils
import wandb

from typing import Any
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from matplotlib.rcsetup import validate_backend

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

def eval_few_shot(model, dataloader, dataloader_test, protos, projected):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proto = []
    min_dists = []
    max_sims = []
    model.eval()
    model.to(device)
    outs = torch.Tensor().to(device)
    outs_train = torch.Tensor().to(device)

    labels = torch.Tensor().to(device)
    labels_train = torch.Tensor().to(device)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for data, data_2, label in dataloader:
        torch.no_grad()
        if projected == False:
            model.backbone_1.fc.register_forward_hook(get_activation("fc_1"))
            model.backbone_2.fc.register_forward_hook(get_activation("fc_2"))
            outs_proj = model(data.float(),data_2.float()).squeeze()

            a = activation["fc_1"]
            b = activation["fc_2"]
            features = torch.cat((a, b), 1)
            outs_train = torch.cat((outs_train, features), 0)
            labels_train = torch.cat((labels_train, label), 0)
    labels_train = labels_train.detach().cpu().repeat(2).numpy()
    outs_train = outs_train.detach().cpu()
    feature_size = 0.5*outs_train.size()[1]
    outs_train = torch.cat((outs_train[:, :int(feature_size)], outs_train[:, int(feature_size):]), 0).detach().cpu().numpy() 
    
    if protos is None:
        for label in np.unique(labels_train):
              proto.append(np.mean(outs_train[labels_train == label], 0))
    else:
        proto = protos

    for data, data_2, label in dataloader_test:
        torch.no_grad()
        if projected == False:
            model.backbone_1.fc.register_forward_hook(get_activation("fc_1"))
            model.backbone_2.fc.register_forward_hook(get_activation("fc_2"))
            outs_proj = model(data.float(),data_2.float()).squeeze()

            a = activation["fc_1"]
            b = activation["fc_2"]
            features = torch.cat((a, b), 1)
            outs = torch.cat((outs, features), 0)
            labels = torch.cat((labels, label), 0)

    labels = labels.detach().cpu().repeat(2).numpy()
    outs = outs.detach().cpu()
    feature_size = 0.5*outs.size()[1]
    outs = torch.cat((outs[:, :int(feature_size)], outs[:, int(feature_size):]), 0)
    count = 0
    correct = 0
    count_sim = 0
    correct_sim = 0

    for i, instance in enumerate(outs):
        distances = np.sum((proto.detach().cpu().numpy() - instance.detach().cpu().numpy())**2, 1)
        min_dist = np.argmin(distances)
        min_dists.append(min_dist)
        if min_dist == labels[i]:
            correct += 1
        count += 1
    
    for i, instance in enumerate(outs):
        cos = torch.nn.CosineSimilarity(dim = 1)
        sim = cos(proto, instance)
        max_sim = torch.argmax(sim)
        max_sims.append(max_sim.detach().cpu().numpy())
        if max_sim == labels[i]:
            correct_sim += 1
        count_sim += 1

    acc_euclid = correct / count
    acc_sim = correct_sim /count_sim
    return acc_euclid, acc_sim

def cal_linclf_acc(model: torch.nn.Module = None, 
                  train_dataloader: torch.utils.data.DataLoader = None,
                  test_dataloader: torch.utils.data.DataLoader = None,
                  num_heads: int = 2,
                  head_no: int = 1,
                  clfs: dict = {
                       "KNeighbors": KNeighborsClassifier(),
                       "NearestCentroid": NearestCentroid(),
                       "SVC": SVC(gamma = "auto")
                       }, 
                  train_sz: float = 0.8,
                  projected: bool = True,
                  protos: torch.Tensor = None,
                  wandb_run: Any = wandb.run,
                  path: str = None,
                  ) -> [Any]:

    utils.set_parameter_requires_grad(model, False)
    if projected == True:
        assert num_heads == 2, \
        "When projection == True, both heads must be used (num_heads == 2)"

    #Calculate embeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    outs = torch.Tensor().to(device)
    #outs_train = torch.Tensor().to(device)
    outs_test = torch.Tensor().to(device)

    labels = torch.Tensor().to(device)
    #labels_train = torch.Tensor().to(device)
    labels_test = torch.Tensor().to(device)

    if num_heads ==2:
        if projected == True:
            for data, data_2, label in train_dataloader:
                torch.no_grad()
                outs = torch.cat((outs, model(data.float(),data_2.float()).squeeze()), 0)
                labels = torch.cat((labels, label), 0)

            for data, data_2, label in test_dataloader:
                torch.no_grad()
                outs_test = torch.cat((outs_test, model(data.float(),data_2.float()).squeeze()), 0)
                labels_test = torch.cat((labels_test, label), 0)
            
        else:
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook

            for data, data_2, label in train_dataloader:
                torch.no_grad()
                model.backbone_1.fc.register_forward_hook(get_activation("fc_1"))
                model.backbone_2.fc.register_forward_hook(get_activation("fc_2"))
                outs_proj = model(data.float(), data_2.float()).squeeze()

                a = activation["fc_1"]
                b = activation["fc_2"]
                features = torch.cat((a, b), 1)
                outs = torch.cat((outs, features), 0)
                labels = torch.cat((labels, label), 0)
            
            for data, data_2, label in test_dataloader:
                torch.no_grad()
                model.backbone_1.fc.register_forward_hook(get_activation("fc_1"))
                model.backbone_2.fc.register_forward_hook(get_activation("fc_2"))
                outs_proj = model(data.float(),data_2.float()).squeeze()

                a = activation["fc_1"]
                b = activation["fc_2"]
                features = torch.cat((a, b), 1)
                outs_test = torch.cat((outs_test, features), 0)
                labels_test = torch.cat((labels_test, label), 0)

        feature_size = 0.5*outs.size()[1]

        outs_cpu = outs.detach().cpu()
        outs_cpu = torch.cat((outs_cpu[:, :int(feature_size)], outs_cpu[:, int(feature_size):]), 0).detach().cpu().numpy()

        outs_test_cpu = outs_test.detach().cpu()
        outs_test_cpu = torch.cat((outs_test_cpu[:, :int(feature_size)], outs_test_cpu[:, int(feature_size):]), 0).detach().cpu().numpy() 

        labels_cpu = labels.detach().cpu().repeat(2).numpy()
        labels_test_cpu = labels_test.detach().cpu().repeat(2).numpy()


    else:
        if head_no == 1:
            model_1 = model.backbone_1
        else:
            model_1 = model.backbone_2

        for data, data_2, label in train_dataloader:
            torch.no_grad()

            outs = torch.cat((outs, model_1(torch.cat((data.float(), data_2.float()))).squeeze()), 0)
            labels = torch.cat((labels, label.repeat(2)), 0)

        for data, data_2, label in test_dataloader:
            torch.no_grad()

            outs_test = torch.cat((outs_test, model_1(torch.cat((data.float(),data_2.float()))).squeeze()), 0)
            labels_test = torch.cat((labels_test, label.repeat(2)), 0)

        outs_cpu = outs.detach().cpu().numpy()
        outs_test_cpu = outs_test.detach().cpu().numpy()

        labels_cpu = labels.detach().cpu().numpy()
        labels_test_cpu = labels_test.detach().cpu().numpy()

    # One Hot Encoder for ROC-AUC measure
    ohe = OneHotEncoder()
    ohe.fit(labels_cpu.reshape(-1, 1))

    # Define Prototypes, if not available (mean)
    proto = []

    if protos is None:
        for label in np.unique(labels_cpu):
              proto.append(np.mean(outs_cpu[labels_cpu == label], 0))
        proto = np.array(proto)
    else:
        proto = protos

    # Prototypical Proximity Evaluation
    ## Euclid Distance Loss
    count, correct = 0, 0
    min_dists = []

    for i, instance in enumerate(outs_test_cpu):
        if protos is not None:
            distances = np.sum((proto.detach().cpu().numpy() - instance)**2, 1)
        else:
            distances = np.sum((proto - instance)**2, 1)

        min_dist = np.argmin(distances)
        min_dists.append(min_dist)

        if min_dist == labels_test_cpu[i]:
            correct += 1
        count += 1

    ## Cosine Sim Loss
    count_sim, correct_sim = 0, 0
    max_sims = []

    for i, instance in enumerate(outs_test_cpu):
        cos = torch.nn.CosineSimilarity(dim = 1)

        if protos is not None:
            sim = cos(proto.detach().cpu(), torch.Tensor(instance))
        else:
            sim = cos(torch.Tensor(proto), torch.Tensor(instance))

        max_sim = torch.argmax(sim)
        max_sims.append(max_sim.detach().cpu().numpy())

        if max_sim == labels_test_cpu[i]:
            correct_sim += 1
        count_sim += 1

    acc_euclid = correct / count
    acc_sim = correct_sim /count_sim

    prototypical_loss = {"Acc_euclid": acc_euclid, "Acc_sim": acc_sim}
    
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
          name_acc = "Lin_clf_acc | " + k
          name_roc = "Lin_clf_ROC | " + k
          if wandb_run is not None:
              wandb_run.summary[name_acc] = acc
              wandb_run.summary[name_roc] = roc

        print("Prototypical Losses")
        for k, v in prototypical_loss.items():
            print(f"Metric: {k} | Value: {v}", file = f)
            print(
              "------------------------------------------------------------------------------------", 
              file = f
              )
            if wandb_run is not None:
                name_acc = "Proto_acc | " + k
                wandb_run.summary[name_acc] = v

def resnet_eval(resnet18_model: torch.nn.Module, 
                dataloader_test: list, 
                num_classes: int, 
                loss_fn: str,
                protos: torch.Tensor,
                path:str) -> list:
    """
    Evaluates ResNet18 model on classification task
    Args:
        resnet18_model: torch.nn.Module: Model to evaluate
        dataloader_test: list(torch.utils.data.Dataloader): 
            List of dataloaders for continual learning setting, i.e., providing
            instances of classes sequentially and ordered
        num_classes: int: total number of classes in the Dataset
        loss_fn: str: which loss function is used loss_fn = {"energy","entropy"}
        path: str: path in which to write the evaluation results to
    Returns:
        list: classification accuracies per class
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corrects = 0
    total = 0
    corrects_list = [0. for i in range(num_classes)]
    counts_list = [0. for i in range(num_classes)]
    test_acc_list = []

    resnet18_model.eval()
    utils.set_parameter_requires_grad(resnet18_model, False)
    resnet18_model.to(device)
    labels_all = torch.Tensor().to(device)
    preds_all = torch.Tensor().to(device)

    for data_1, data_2, label in dataloader_test:
        torch.no_grad()
        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        labels = label.to(device)
        outs_1 = resnet18_model(data_1)
        outs_2 = resnet18_model(data_2)
        outs = torch.cat((outs_1, outs_2), 0).to(device)

        labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1).squeeze()
        
        if loss_fn == "entropy":
            _, preds = torch.max(outs, 1)

        elif loss_fn == "energy":
            _, preds = torch.min(outs, 1)

        elif loss_fn == "proto_vicreg":
            protos = protos.detach()
            dists = torch.Tensor().to(device)
            for proto in protos:
                mse = F.mse_loss(outs, 
                                proto.repeat(outs.size()[0], 1), 
                                reduction = "none").sum(dim = 1, keepdim = True)
                dists = torch.cat((dists, mse), 1)

            _, preds = torch.min(dists, 1)        


        labels_all = torch.cat((labels_all, labels))
        preds_all = torch.cat(((preds_all, preds)))
        
        for i, label in enumerate(range(num_classes)):
            corrects = 0
            total = 0

            mask = label == labels.detach().cpu().numpy()
            preds_masked = preds[mask].detach().cpu().numpy()
            labels_masked = labels[mask].detach().cpu().numpy()

            corrects_list[i] += (preds_masked == labels_masked).sum()
            counts_list[i] += labels_masked.size

    for i, j in zip(corrects_list, counts_list):
        test_acc_list.append(i/j)
    labels_all, preds_all = labels_all.detach().cpu().numpy(), preds_all.detach().cpu().numpy()

    # with open(path, "a") as f:
    #     print(classification_report(labels_all, preds_all), file = f)
        # Sanity check
        # for i, entry in enumerate(test_acc_list):
        #     print(f"Class: {i} | Test Acc: {entry}", file = f)
        # print(f"Overall Test Acc: {np.sum(test_acc_list)/num_classes:.2}", file = f)
    test_acc = np.mean(test_acc_list)
    for i, entry in enumerate(test_acc_list):
        print(f"Class: {i} | Test Acc: {entry}")
    print(f"Overall Test Acc: {test_acc:.2}")
    return test_acc, test_acc_list

# Get predcition for eval
def get_prediction(outs, labels, outs_storage, label_storage, protos):
    dists = torch.Tensor().to(device)
    outs_total = torch.cat((outs, outs_storage))
    labels_total = torch.cat((labels, label_storage))
    for proto in protos:
        mse = F.mse_loss(outs_total, 
                        proto.repeat(outs_total.size()[0], 1), 
                        reduction = "none").sum(dim = 1, keepdim = True)
        dists = torch.cat((dists, mse), 1)
    _, preds = torch.min(dists, 1)
    return preds, labels_total

# Calculate Forgetting Metric
def forgetting_metric(test_acc_lists, num_classes, total_classes, cls_per_run):
    acc_array = np.zeros(shape = (len(test_acc_lists), total_classes))
    for i, epoch_list in enumerate(test_acc_lists):
        for j, entry  in enumerate(epoch_list):
            acc_array[i, j] = entry
    forgetting_per_cls = []

    for i in range(acc_array.shape[1]):
        if np.max(acc_array[:, i]) > 0:
            forgetting_per_cls.append(np.max(acc_array[:, i]) - acc_array[-1, i]/np.max(acc_array[:, i]))
        else:
            forgetting_per_cls.append(0)            
    previous_cls = num_classes - cls_per_run
    
    if previous_cls > 0:
        forgetting_metric = np.mean(forgetting_per_cls[:previous_cls])
    else: 
        forgetting_metric = 0
    return forgetting_metric, forgetting_per_cls