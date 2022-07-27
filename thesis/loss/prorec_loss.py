import torch.nn as nn

import torch
import torch.nn.functional as F

import copy

def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1))
    return std_loss


def covariance_loss(z1: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: covariance regularization loss.
    """
    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D
    return cov_loss

def proto_vicreg_loss_func(
    model: torch.nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    act: torch.Tensor,
    protos: torch.Tensor,
    protos_act: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
    dist_loss_weight: float = 1.0,
    proto_reg_loss_weight: float = 0.0,
    outs_storage: torch.Tensor = None,
    label_storage: torch.Tensor = None,
    min_cls_sampling: int = 128
) -> torch.Tensor:
    """Computes VICReg's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.
    Returns:
        torch.Tensor: VICReg loss.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_loss = torch.Tensor([0]).to(device)
    dist_loss = torch.Tensor([0]).to(device)
    proto_reg_loss = torch.Tensor([0]).to(device)
    prototypes = torch.Tensor().to(device)
    prototypes_act = torch.Tensor().to(device)
    ood_storage = torch.Tensor().to(device)

    #protos = torch.Tensor().to(device)
    # linear = copy.deepcopy(model.fc)

    if protos_act.size()[0] > 0:
        protos_act_new = model.fc(protos_act)

    if outs_storage.size()[0] > 0:
        z_storage = model.fc(outs_storage)
        z_ext = torch.cat((z, z_storage), 0)
        labels_ext = torch.cat((labels, label_storage))
    else:
        z_ext = z
        labels_ext = labels

    # -> Define Prototypes
    for label in range(num_classes):
        mask = label == labels.squeeze()

        mask_ext = label == labels_ext.squeeze()

        # Allocate prototypes 
        # Proto for given label is existent & there are examples of label  in batch
        if (label < protos.size()[0]) & (mask.sum() > 0):
            prototypes = torch.cat((prototypes, z[mask].mean(dim = 0, keepdim = True)), 0)
            prototypes_act = torch.cat((prototypes_act, act[mask].mean(dim = 0, keepdim = True)), 0)
            # proto_reg_loss += invariance_loss(protos[label], prototypes[label])

        # Proto for given label is existent & there are examples only in storage
        elif (label < protos.size()[0]) & (mask_ext.sum() > 0):
            prototypes = torch.cat((prototypes, z_ext[mask_ext].mean(dim = 0, keepdim = True)), 0)
            prototypes_act = torch.cat((prototypes_act, act[mask].mean(dim = 0, keepdim = True)), 0)
            # Regularize proto from storage to limit changes
            proto_reg_loss += invariance_loss(protos[label], prototypes[label])

        # Proto for given label is existent & there are no examples in batch
        # Take existent proto
        elif (label < protos.size()[0]) & (mask.sum() == 0) & (mask_ext.sum() == 0):
            prototypes = torch.cat((prototypes, protos_act_new[label].unsqueeze(dim = 0)), 0)
            prototypes_act = torch.cat((prototypes_act, protos_act[label].unsqueeze(dim = 0)), 0)
            proto_reg_loss += invariance_loss(protos[label], prototypes[label])

        # No proto for given label yet existent, but examples in batch
        # Creates new proto for label
        elif (mask.sum() > 0):
            prototypes = torch.cat((prototypes, z[mask].mean(dim = 0, keepdim = True)), 0)
            prototypes_act = torch.cat((prototypes_act, act[mask].mean(dim = 0, keepdim = True)), 0)
            print(f"Creating new proto for class {label}")

    # -> Sim Loss
    for i, _ in enumerate(prototypes):
        if i < prototypes.size()[0] - 1:
            # sim_loss += F.relu(1000 - invariance_loss(
            #                               proto_1.repeat(prototypes[i + 1:].size()[0], 1), 
            #                               prototypes[i + 1:]
            #                               )
            # )
            sim_loss -= invariance_loss(
                                          prototypes[i].repeat(prototypes[i + 1:].size()[0], 1), 
                                          prototypes[i + 1:]
                                          )

        #Attract samples to the prototype that are of label == i
        pos_samples = z_ext[i == labels_ext.squeeze()]
        if ((i == labels_ext.squeeze()).sum() > 0):
            dist_loss += invariance_loss(z_ext[i == labels_ext.squeeze()], 
                                                 prototypes[i].repeat(pos_samples.size()[0], 1)
                                                 )/pos_samples.size()[0]

    # Prevent informational collapse
    cov_loss = covariance_loss(prototypes)
    var_loss = variance_loss(prototypes)

    # Increase distance between prototypes
    # Maintain variance between the variables of the prototypes
    # Increase informational content of each prototype
    loss = sim_loss_weight * torch.exp(sim_loss) \
    + var_loss_weight * var_loss \
    + cov_loss_weight * cov_loss \
    + dist_loss_weight * dist_loss \
    + proto_reg_loss_weight * proto_reg_loss
    print(f"Proto_reg_loss: {proto_reg_loss_weight * proto_reg_loss}")
    return loss.squeeze(), prototypes, prototypes_act