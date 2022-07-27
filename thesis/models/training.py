import os
import torch
from torch import nn as nn
import torch.optim as optim
import torchvision

from tqdm.auto import tqdm
import copy
from sklearn.metrics import classification_report

import wandb

from IPython.display import clear_output

from thesis.loss import vicreg_loss_fn as vlf
from thesis.loss.similarity_loss import cosine_sim
from thesis.loss.similarity_loss import NCELoss
from thesis.loss.energy_loss import energy_logit_loss
from thesis.models.eval import resnet_eval

#############################################################
################# Training Step for GNN #####################
#############################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Train the given model on the given dataset for num_epochs
def train_gat(model, train_loader, test_loader, num_epochs, edge_attr = True):
    """

    Args:
      model: 
      train_loader: 
      test_loader: 
      num_epochs: 
      edge_attr: (Default value = True)

    Returns:

    """

    # Set up the loss and the optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    train_acc_ls = []
    test_acc_ls = []
    loss_ls = []

    # A utility function to compute the accuracy
    def get_train_acc(model, loader):
        """

        Args:
          model: 
          loader: 

        Returns:

        """
        n_total = 0
        n_ok = 0
        with torch.no_grad():
          for data in loader:
              data.to(device)
              outs = model(
                  data.x.float(), 
                  data.edge_index, 
                  data.edge_attr.float(), 
                  data.batch
                  ).float()
              n_ok += (torch.argmax(outs, dim = 1) == data.y).sum().item()
              n_total += data.y.shape[0]
          return n_ok/n_total

    def get_test_acc(model, loader):
        """

        Args:
          model: 
          loader: 

        Returns:

        """
        n_total = 0
        n_ok = 0
        with torch.no_grad():
          for data in loader:
              data.to(device)

              outs = model(
                  data.x.float(), 
                  data.edge_index, 
                  data.edge_attr.float(), 
                  data.batch
                  ).float()
              n_ok += (torch.argmax(outs, dim = 1) == data.y).sum().item()
              n_total += data.y.shape[0]
          return n_ok/n_total   

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for data in tqdm(train_loader, leave = False):
            # Zero grads -> forward pass -> compute loss -> backprop
            data.to(device)
            model.to(device)
            optimizer.zero_grad()

           
            outs = model(
                data.x.float(), 
                data.edge_index, 
                data.edge_attr.float(), 
                data.batch
                ).float().squeeze()
            loss = loss_fn(outs, data.y.long()).float() # no train_mask!
            loss.backward()
            optimizer.step()

        # Compute accuracies
        acc_train = get_train_acc(model.to(device), train_loader)
        acc_test = get_test_acc(model.to(device), test_loader)
        #writer.add_scalar("Loss/train", loss, epoch)
        #writer.add_scalar("Acc/train", acc_train, epoch)

        print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Train: {acc_train:.3f} | Test: {acc_test:.3f}')
        train_acc_ls.append(acc_train)
        test_acc_ls.append(acc_test)
        loss_ls.append(loss)
    return train_acc_ls, test_acc_ls, loss_ls


#################################################################
################# Training Step for GNN PNA #####################
#################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Train the given model on the given dataset for num_epochs
def train_pna(model, train_loader, test_loader, num_epochs, edge_attr = True):
    """

    Args:
      model: 
      train_loader: 
      test_loader: 
      num_epochs: 
      edge_attr: (Default value = True)

    Returns:

    """

    # Set up the loss and the optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_acc_ls = []
    test_acc_ls = []
    loss_ls = []

    # A utility function to compute the accuracy
    def get_train_acc(model, loader):
        """

        Args:
          model: 
          loader: 

        Returns:

        """
        n_total = 0
        n_ok = 0
        with torch.no_grad():
          for data in loader:
              data.to(device)
              outs = model(data.x.float(), data.edge_index, data.batch).float()
              n_ok += (torch.argmax(outs, dim = 1) == data.y).sum().item()
              n_total += data.y.shape[0]
          return n_ok/n_total

    def get_test_acc(model, loader):
        """

        Args:
          model: 
          loader: 

        Returns:

        """
        n_total = 0
        n_ok = 0
        with torch.no_grad():
          for data in loader:
              data.to(device)

              outs = model(data.x.float(), data.edge_index, data.batch).float()
              n_ok += (torch.argmax(outs, dim = 1) == data.y).sum().item()
              n_total += data.y.shape[0]
          return n_ok/n_total   

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for data in tqdm(train_loader, leave = False):
            # Zero grads -> forward pass -> compute loss -> backprop
            data.to(device)
            model.to(device)
            optimizer.zero_grad()
            outs = model(
                data.x.float(), 
                data.edge_index, 
                data.batch
                ).float().squeeze()
            loss = loss_fn(outs, data.y.long()).float() # no train_mask!
            loss.backward()
            optimizer.step()

        # Compute accuracies
        acc_train = get_train_acc(model.to(device), train_loader)
        acc_test = get_test_acc(model.to(device), test_loader)
        #writer.add_scalar("Loss/train", loss, epoch)
        #writer.add_scalar("Acc/train", acc_train, epoch)

        print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Train: {acc_train:.3f} | Test: {acc_test:.3f}')
        train_acc_ls.append(acc_train)
        test_acc_ls.append(acc_test)
        loss_ls.append(loss)
    return train_acc_ls, test_acc_ls, loss_ls

#################################################################
################# Training Step for VICReg  #####################
#################################################################
def train_vicreg(model, train_loader, test_loader, epochs, root_dir = None) -> torch.Tensor:
    """Training step for VICReg reusing BaseMethod training step.

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
      model: 
      train_loader: 
      test_loader: 
      epochs: 
      root_dir: (Default value = None)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.

    """
    model.train()
    model.to(device)
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for epoch in tqdm(range(epochs)):
        batch_count = 0
        PATH = os.path.join(root_dir, f"{epoch}.pt")

        for image_data, graph_data in tqdm(train_loader, leave = False):
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            
            out = model(image_data.float(), graph_data).float().squeeze()
            feature_size = out.size()[1]
            #print(out[:,:int(feature_size*0.5)], out[:, int(feature_size*0.5):])

            vicreg_loss = vlf.vicreg_loss_fn(
                out[:,:int(feature_size*0.5)],
                 out[:, int(feature_size*0.5):]
                 ).float()
            vicreg_loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch} | Batch: {batch_count} | Loss: {vicreg_loss:.3f}")
            batch_count += 1
        loss_list.append(vicreg_loss/batch_count)
        print(f"Epoch loss: {vicreg_loss/batch_count:.2f}")
        batch_count = 0

        if (epoch % 10 == 0) | (epoch == 0):
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': vicreg_loss,
              }, PATH)
    
    return loss_list

#################################################################
################# Training Step for ResNet Backbone  ############
#################################################################

def model_train(model, train_loader, epochs):
    """

    Args:
      model: 
      train_loader: 
      epochs: 

    Returns:

    """
    model.train()
    model.to(device)
    loss_list =  []

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in tqdm(range(epochs)):
      batch_count = 0
      epoch_loss = 0
      batch_acc_list = []
      batch_loss_list = []

      for data, label in tqdm(train_loader, leave = False):
        data.to(device)
        optimizer.zero_grad()

        n_corr = 0
        n_total = 0

        out = model(data.float()).squeeze()
        loss = loss_fn(out, label.long())
        loss.backward()
        optimizer.step()

        n_corr += (torch.argmax(out, dim = 1) == label.long()).sum().float().item()
        n_total += label.size()[0]

        acc = n_corr/n_total
        batch_acc_list.append(acc)
        batch_loss_list.append(loss)
        batch_count += 1
        #print(f"Batch Accuracy: {acc:.2f} | Batch loss: {loss:.2f}")
      epoch_loss = (sum(batch_loss_list)/batch_count)
      epoch_acc = (sum(batch_acc_list)/batch_count)
      print(f"Epoch: {epoch} | Loss: {epoch_loss} | Accuracy: {epoch_acc}")
      loss_list.append(epoch_loss)
    return loss_list

#################################################################
################# Training Step for Graph Backbone  ############
#################################################################

def train_step_graph(model, dataloader, epochs, optimizer = None):
    """

    Args:
      model: 
      dataloader: 
      epochs: 
      optimizer: (Default value = None)

    Returns:

    """
    model.train()
    model.to(device)

    n_correct = 0
    n_total = 0
    loss_list = []
    acc_list = []
    loss_fn = nn.NLLLoss()
    corr_pred = []

    if optimizer == None:
      optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    else:
      optimizer = optimizer

    for epoch in tqdm(range(epochs)):
      for graph in tqdm(dataloader, leave = False):
        graph.to(device)
        optimizer.zero_grad()
        out = model(graph.x.float(), graph.edge_index, graph.edge_attr.float(), graph.batch).float().squeeze()
        loss = loss_fn(out, graph.y.long()).float()
        loss.backward()
        optimizer.step()

        n_correct += (torch.argmax(out, dim = 1) == graph.y).sum().item()
        n_total += graph.y.shape[0]
        acc = n_correct/n_total
        print(f"Batch accuracy: {acc}")

      print(f"Epoch: {epoch} | Epoch loss: {loss} | Accuracy: {acc}")
      acc_list.append(acc)
      loss_list.append(loss)
    return loss_list, acc_list

################################################################################
################# Training Step for VICReg ResNet x Graph x Semi ###############
################################################################################

def train_vicreg_graph_semi(
  model, 
  train_loader, 
  test_loader, 
  epochs,
  weight_vicreg = 1, 
  weight_sim = 1,
  lr = 0.001,
  t = 0.3,
  root_dir = None) -> torch.Tensor:

    """Training step for Self-Supervised Training model with VICReg and Sim loss

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
      model: 
      train_loader: 
      test_loader: 
      epochs: 
      weight_vicreg: (Default value = 1)
      weight_sim: (Default value = 1)
      lr: (Default value = 0.001)
      t: (Default value = 0.3)
      root_dir: (Default value = None)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.
      Gratefully adapted from: https://github.com/vturrisi/solo-learn

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

    for epoch in tqdm(range(epochs)):
        batch_count = 0
        batch_loss = 0
        epoch_loss = 0
        counter = 0

        PATH = os.path.join(root_dir, f"{epoch}.pt")

        for image_data, graph_data in tqdm(train_loader, leave = False):
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            
            out = model(image_data.float(), graph_data).float().squeeze()
            feature_size = out.size()[1]

            labels = graph_data.y.view(graph_data.y.size(dim = 0), 1).repeat(2, 1)
            loss = weight_vicreg*vlf.vicreg_loss_func(
                out[:,:int(feature_size*0.5)],
                out[:, int(feature_size*0.5):],
                ).float() + weight_sim* NCELoss(
                    torch.cat(
                        (
                            out[:,:int(feature_size*0.5)], out[:, int(feature_size*0.5):]
                          ), dim = 0), 
                          labels,
                          t,
                ).float()

                    
            loss.backward()
            optimizer.step()

            batch_count += 1
            batch_loss += loss
            print(f"Epoch: {epoch} | Loss: {loss.detach().cpu().numpy()}")
        
        clear_output()
        epoch_loss = batch_loss/batch_count
        loss_list.append(epoch_loss.detach().cpu().numpy())
        print(f"Epoch: {epoch} |Epoch loss: {(epoch_loss):.2f}")
        #ax.plot(loss_list)
        #plt.show()

        batch_count = 0
        batch_loss = 0

        if epoch_loss.detach().cpu().numpy() >= min(loss_list):
          counter += 1
        else:
          counter = 0

        if counter > 5:
          return loss_list
          break

        if epoch % 500 == 0:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss,
              }, PATH)
    
    return loss_list


################################################################################
################# Training Step for VICReg ResNet x ResNet x Semi ##############
################################################################################
import os
import torch
import numpy as np
from torch import nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from IPython.display import clear_output

from thesis.loss import vicreg_loss_fn as vlf
from thesis.loss.similarity_loss import cosine_sim
from thesis.loss.similarity_loss import NCELoss

def train_vicreg_cnn_2(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  epochs: int,
  weight_vicreg: float = 1,
  sim_vicreg: float = 25,
  var_vicreg: float = 25,
  cov_vicreg: float = 1,
  decay_rate_vicreg: float = 0.01,
  decay_steps_vicreg: float = 100,
  weight_sim: float = 0.01,
  decay_rate_sim: float = 0.01,
  decay_steps_sim: float = -100,
  lr: float = 0.001,
  t: float = 1.,
  alpha: float = 0.5,
  alpha_prot: float = 0.3,
  epsilon: float = 0.05,
  instance_weight: float = 1,
  proto_weight: float = 5,
  cel_weight: float = 1,
  dist_weight: float = 500,
  num_classes: float = 3,
  sim_loss_fn: str = "cosine",
  lr_scheduler: str = "exp",
  gamma: float = 0.9,
  ssv_prob: float = 1,
  root_dir = None, **kwargs) -> torch.Tensor:

    """Training step for Self-Supervised Training model with VICReg and Supervised
    Training with Sim loss.

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
 
      model: torch.nn.Module: 
      dataloader: torch.utils.data.DataLoader: 
      epochs: int: 
      weight_vicreg: float:  (Default value = 1)
      sim_vicreg: float:  (Default value = 25)
      var_vicreg: float:  (Default value = 25)
      cov_vicreg: float:  (Default value = 1)
      decay_rate_vicreg: float:  (Default value = 0.01)
      decay_steps_vicreg: float:  (Default value = 100)
      weight_sim: float:  (Default value = 0.01)
      decay_rate_sim: float:  (Default value = 0.01)
      decay_steps_sim: float:  (Default value = -100)
      lr: float:  (Default value = 0.001)
      t: float:  (Default value = 1.)
      alpha: float:  (Default value = 0.5)
      alpha_prot: float:  (Default value = 0.3)
      epsilon: float:  (Default value = 0.05)
      instance_weight: float:  (Default value = 1)
      proto_weight: float:  (Default value = 5)
      cel_weight: float:  (Default value = 1)
      dist_weight: float:  (Default value = 500)
      num_classes: float:  (Default value = 3)
      sim_loss_fn: str:  (Default value = "cosine")
      lr_scheduler: str:  (Default value = "exp")
      gamma: float:  (Default value = 0.9)
      ssv_prob: float:  (Default value = 1)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.
      Gratefully adapted from: https://github.com/vturrisi/solo-learn

    """

    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model.train()
      model.to(device)

      # Initiate return variables
      loss_list, vicreg_loss_list, sim_loss_list, prototypes_list = [], [], [], []
      prototypes = None
      
      # Define optimizer and scheduler
      optimizer = torch.optim.Adam(model.parameters(), lr = lr)
      if lr_scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

      weight_vicreg_init = weight_vicreg
      weight_sim_init = weight_sim

      # Training loop
      for epoch in tqdm(range(epochs)):
          batch_count = batch_loss = vicreg_batch_loss = sim_batch_loss = epoch_loss = 0
          
          # VICReg Loss Decay
          if (bool(decay_rate_vicreg)) & (epoch <= abs(decay_steps_vicreg)):
              weight_vicreg = weight_vicreg_init*decay_rate_vicreg**(epoch/decay_steps_vicreg)
          
          # Sim Loss Decay
          if (bool(decay_rate_sim)) & (epoch <= abs(decay_steps_sim)):
              weight_sim = weight_sim_init*decay_rate_sim**(epoch/decay_steps_sim)

          # Batch loop
          for image_1, image_2, labels in tqdm(dataloader, leave = False):
              
              # Zero grads -> forward pass -> compute loss -> backprop
              optimizer.zero_grad()
              out = model(image_1.float(), image_2.float()).float().squeeze()
              feature_size = out.size()[1]

              labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1)
              
              # Calculate VICReg Loss function
              vicreg_loss = weight_vicreg*vlf.vicreg_loss_func(
                  out[:,:int(feature_size*0.5)],
                  out[:, int(feature_size*0.5):], sim_loss_weight = sim_vicreg,
                  var_loss_weight = var_vicreg, cov_loss_weight = cov_vicreg,
                  ).float()
              
              # Assign features for handling in following loss functions from model
              # outputs
              sim_features = torch.cat(
                              (
                                  out[:,:int(feature_size*0.5)],
                                  out[:, int(feature_size*0.5):]
                                ), dim = 0)
              
              # Implementation of cosine similarity loss
              if sim_loss_fn == "cosine":
                  sim_loss = weight_sim*cosine_sim(
                      sim_features, labels, t, alpha
                      ).float()
              
              #Implementation of NCELoss
              elif sim_loss_fn == "NCELoss":
                  sim_loss = weight_sim*NCELoss(sim_features, labels, t).float()

              #Implementation of prototypical loss
              elif sim_loss_fn == "proto":        
                  sim_loss, instance_loss, proto_loss, \
                  ce_loss, dist_loss, prototypes_updated = proto_sim(
                                reps = sim_features, labels = labels, 
                                prototypes = prototypes, 
                                t = t, alpha = alpha, alpha_prot = alpha_prot, 
                                instance_weight = instance_weight, 
                                proto_weight = proto_weight, dist_weight = dist_weight,
                                cel_weight = cel_weight, num_classes = num_classes,
                                epsilon = epsilon, epoch = epoch
                                )
                  
                  # Reassign prototypes
                  prototypes = prototypes_updated.detach()
                  sim_loss = weight_sim*sim_loss.float().detach()
              
              # Determine the probability with which supervised labels will be used
              semi_sup_ = int(np.random.choice(2, 1, p = [1- ssv_prob, ssv_prob]))
              loss = vicreg_loss + semi_sup_*sim_loss

              loss.backward()
              optimizer.step()

              # Output batch losses
              batch_count += 1
              batch_loss += loss.detach().cpu().numpy()
              vicreg_batch_loss += vicreg_loss.detach().cpu().numpy()
              sim_batch_loss += sim_loss.detach().cpu().numpy()
              print(f"Epoch: {epoch} | Batch_Loss: {loss.detach().cpu().numpy()}")
              
          clear_output()

          # Calculate and log epoch losses
          epoch_loss = batch_loss/batch_count
          vicreg_loss = vicreg_batch_loss/batch_count
          sim_loss = sim_batch_loss/batch_count

          if sim_loss_fn == "proto":
              wandb.log({
                          "loss":epoch_loss, 
                          "vicreg_loss": vicreg_loss, 
                          "sim_loss": sim_loss,
                          "sim_loss_norm": sim_loss/weight_sim,
                          "vicreg_loss_norm": vicreg_loss/weight_vicreg,
                          "weight_vicreg":weight_vicreg, 
                          "weight_sim":weight_sim,
                          "inst_loss":instance_loss,
                          "proto_loss":proto_loss,
                          "ce_loss":ce_loss,
                          "dist_loss":dist_loss,
                          "prototypes":prototypes_updated,
                        })
              prototypes_list.append(prototypes_updated.detach().cpu().numpy())

              
          elif sim_loss_fn == "cosine":
              wandb.log({
                          "loss":epoch_loss, 
                          "vicreg_loss": vicreg_loss, 
                          "sim_loss": sim_loss,
                          "sim_loss_norm": sim_loss/weight_sim,
                          "vicreg_loss_norm": vicreg_loss/weight_vicreg,
                          "weight_vicreg":weight_vicreg, 
                          "weight_sim":weight_sim,
                        })              
          loss_list.append(epoch_loss)
          vicreg_loss_list.append(vicreg_loss)
          sim_loss_list.append(sim_loss)

          print(f"Epoch: {epoch} | Epoch loss: {epoch_loss:.2f}")
            
          # Save model, in case a root_dir is given
          if (epoch > 5) & (root_dir is not None) & (~np.isnan(loss.detach().cpu().numpy())):
              if (loss < loss_list[-2]):
                  PATH = os.path.join(root_dir, f"{run_name}.pt")
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss,
                    }, PATH) 
        
            
      # Return loss logs and prototypes, in case it's given
      if sim_loss_fn == "proto":
          return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list
      else:
          return loss_list, vicreg_loss_list, sim_loss_list

    except KeyboardInterrupt:
        print("Execution interrupted by user")
        if sim_loss_fn == "proto":
            return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list
        else:
            return loss_list, vicreg_loss_list, sim_loss_list

################################################################################
####################### Training scheme VicReg CNNx2 and energy loss ###########
################################################################################


def train_vic_cnn_2_enloss(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  epochs: int,
  weight_vicreg: float = 1,
  sim_vicreg: float = 25,
  var_vicreg: float = 25,
  cov_vicreg: float = 1,
  decay_rate_vicreg: float = 0.01,
  decay_steps_vicreg: float = 100,
  weight_sim: float = 0.01,
  decay_rate_sim: float = 0.01,
  decay_steps_sim: float = -100,
  lr: float = 0.001,
  t: float = 1.,
  m: float = 0.5,
  alpha: float = 0.5,
  lr_scheduler: str = "exp",
  metric: str = "euclid",
  warm_up: int = 20,
  num_classes: int = 3,
  gamma: float = 0.9,
  root_dir: str = None,
  run_name: str = None, 
  **kwargs) -> torch.Tensor:

    """Training step for Self-Supervised Training model with VICReg and Sim loss

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
      model: torch.nn.Module:
      dataloader: torch.utils.data.DataLoader:
      epochs: int:
      weight_vicreg: float:  (Default value = 1)
      sim_vicreg: float:  (Default value = 25)
      var_vicreg: float:  (Default value = 25)
      cov_vicreg: float:  (Default value = 1)
      decay_rate_vicreg: float:  (Default value = 0.01)
      decay_steps_vicreg: float:  (Default value = 100)
      weight_sim: float:  (Default value = 0.01)
      decay_rate_sim: float:  (Default value = 0.01)
      decay_steps_sim: float:  (Default value = -100)
      lr: float:  (Default value = 0.001)
      t: float:  (Default value = 1.)
      m: float:  (Default value = 0.5)
      alpha: float:  (Default value = 0.5)
      lr_scheduler: str:  (Default value = "exp")
      metric: str:  (Default value = "euclid")
      warm_up: int:  (Default value = 20)
      num_classes: int:  (Default value = 3)
      gamma: float:  (Default value = 0.9)
      root_dir: str:  (Default value = None)
      run_name: str:  (Default value = None)
      **kwargs: 
      model: torch.nn.Module: 
      dataloader: torch.utils.data.DataLoader: 
      epochs: int: 
      weight_vicreg: float:  (Default value = 1)
      sim_vicreg: float:  (Default value = 25)
      var_vicreg: float:  (Default value = 25)
      cov_vicreg: float:  (Default value = 1)
      decay_rate_vicreg: float:  (Default value = 0.01)
      decay_steps_vicreg: float:  (Default value = 100)
      weight_sim: float:  (Default value = 0.01)
      decay_rate_sim: float:  (Default value = 0.01)
      decay_steps_sim: float:  (Default value = -100)
      lr: float:  (Default value = 0.001)
      t: float:  (Default value = 1.)
      m: float:  (Default value = 0.5)
      alpha: float:  (Default value = 0.5)
      lr_scheduler: str:  (Default value = "exp")
      metric: str:  (Default value = "euclid")
      warm_up: int:  (Default value = 20)
      num_classes: int:  (Default value = 3)
      gamma: float:  (Default value = 0.9)
      root_dir: str:  (Default value = None)
      run_name: str:  (Default value = None)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.
      Gratefully adapted from: https://github.com/vturrisi/solo-learn

    """

    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model.train()
      model.to(device)
      prototypes = torch.Tensor().to(device)
      counts = torch.Tensor().to(device)

      # Initiate return variables
      loss_list, vicreg_loss_list, sim_loss_list, prototypes_list = [], [], [], []
      
      # Define optimizer and scheduler
      optimizer = torch.optim.Adam(model.parameters(), lr = lr)

      if lr_scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

      weight_vicreg_init = weight_vicreg
      weight_sim_init = weight_sim

      # Training loop
      for epoch in tqdm(range(epochs)):
          batch_count = batch_loss = vicreg_batch_loss = sim_batch_loss = epoch_loss = 0
          
          # VICReg Loss Decay
          if (bool(decay_rate_vicreg)) & (epoch <= abs(decay_steps_vicreg)):
              weight_vicreg = weight_vicreg_init*decay_rate_vicreg**(epoch/decay_steps_vicreg)
          
          # Sim Loss Decay
          if (bool(decay_rate_sim)) & (epoch <= abs(decay_steps_sim)):
              weight_sim = weight_sim_init*decay_rate_sim**(epoch/decay_steps_sim)

          # Batch loop
          for image_1, image_2, labels in tqdm(dataloader, leave = False):
              
              # Zero grads -> forward pass -> compute loss -> backprop
              optimizer.zero_grad()
              out = model(image_1.float(), image_2.float()).float().squeeze()
              feature_size = out.size()[1]
              labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1)
              
              # Calculate VICReg Loss function
              vicreg_loss = weight_vicreg*vlf.vicreg_loss_func(
                  out[:,:int(feature_size*0.5)],
                  out[:, int(feature_size*0.5):], sim_loss_weight = sim_vicreg,
                  var_loss_weight = var_vicreg, cov_loss_weight = cov_vicreg,
                  ).float()
              
              # Assign features for handling in following loss functions from model
              # outputs
              sim_features = torch.cat(
                              (
                                  out[:,:int(feature_size*0.5)],
                                  out[:, int(feature_size*0.5):]
                                ), dim = 0)
              
              #Implementation of prototypical loss
              sim_loss, prototypes, counts = energy_loss(
                  reps = sim_features, 
                  labels = labels,
                  prototypes = prototypes,
                  alpha = alpha,
                  metric = metric,
                  warm_up = warm_up,
                  epoch = epoch,
                  counts = counts,
                  t = t,
                  m = m,
                  )
              
              # Reassign prototypes
              #prototypes = prototypes.detach()
              sim_loss = weight_sim*sim_loss.float().detach()
              
              # Determine the probability with which supervised labels will be used
              loss = vicreg_loss + sim_loss
              loss.backward()
              optimizer.step()

              # Output batch losses
              batch_count += 1
              batch_loss += loss.detach().cpu().numpy()
              vicreg_batch_loss += vicreg_loss.detach().cpu().numpy()
              sim_batch_loss += sim_loss.detach().cpu().numpy()
              print(f"Epoch: {epoch} | Batch_Loss: {loss.detach().cpu().numpy()}")
              
          clear_output()

          # Calculate and log epoch losses
          epoch_loss = batch_loss/batch_count
          vicreg_loss = vicreg_batch_loss/batch_count
          sim_loss = sim_batch_loss/batch_count

          wandb.log({
                      "loss":epoch_loss, 
                      "vicreg_loss": vicreg_loss, 
                      "sim_loss": sim_loss,
                      "sim_loss_norm": sim_loss/weight_sim,
                      "vicreg_loss_norm": vicreg_loss/weight_vicreg,
                      "weight_vicreg":weight_vicreg, 
                      "weight_sim":weight_sim,
                      "prototypes":prototypes.detach(),
                    })
          
          prototypes_list.append(prototypes.detach().cpu().numpy())
              
          loss_list.append(epoch_loss)
          vicreg_loss_list.append(vicreg_loss)
          sim_loss_list.append(sim_loss)

          print(f"Epoch: {epoch} | Epoch loss: {epoch_loss:.2f}")

          # Save model, in case a root_dir is given
          if (root_dir != None) & (loss < loss_list[-1]):
            PATH = os.path.join(root_dir, f"{run_name}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)          
            
      # Return loss logs and prototypes, in case it's given
      return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list

    except KeyboardInterrupt:
        print("Execution interrupted by user")
        return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list

import torchvision
import torch.nn as nn
import copy
from sklearn.metrics import classification_report

import os
import torch
from torch import nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import wandb

from IPython.display import clear_output

from thesis.loss import vicreg_loss_fn as vlf
from thesis.loss.similarity_loss import cosine_sim
from thesis.loss.similarity_loss import NCELoss

from sklearn.metrics import classification_report
from thesis.models.eval import resnet18_eval

#############################################################
################# Training Step for GNN #####################
#############################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#################################################################
################# Training Step for VICReg  #####################
#################################################################
def train_vicreg(model, train_loader, test_loader, epochs, root_dir = None) -> torch.Tensor:
    """Training step for VICReg reusing BaseMethod training step.

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
      model: 
      train_loader: 
      test_loader: 
      epochs: 
      root_dir: (Default value = None)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.

    """
    model.train()
    model.to(device)
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for epoch in tqdm(range(epochs)):
        batch_count = 0
        PATH = os.path.join(root_dir, f"{epoch}.pt")

        for image_data, graph_data in tqdm(train_loader, leave = False):
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            
            out = model(image_data.float(), graph_data).float().squeeze()
            feature_size = out.size()[1]
            #print(out[:,:int(feature_size*0.5)], out[:, int(feature_size*0.5):])

            vicreg_loss = vlf.vicreg_loss_fn(
                out[:,:int(feature_size*0.5)],
                 out[:, int(feature_size*0.5):]
                 ).float()
            vicreg_loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch} | Batch: {batch_count} | Loss: {vicreg_loss:.3f}")
            batch_count += 1
        loss_list.append(vicreg_loss/batch_count)
        print(f"Epoch loss: {vicreg_loss/batch_count:.2f}")
        batch_count = 0

        if (epoch % 10 == 0) | (epoch == 0):
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': vicreg_loss,
              }, PATH)
    
    return loss_list

#################################################################
################# Training Step for ResNet Backbone  ############
#################################################################

def model_train(model, train_loader, epochs):
    """

    Args:
      model: 
      train_loader: 
      epochs: 

    Returns:

    """
    model.train()
    model.to(device)
    loss_list =  []

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in tqdm(range(epochs)):
      batch_count = 0
      epoch_loss = 0
      batch_acc_list = []
      batch_loss_list = []

      for data, label in tqdm(train_loader, leave = False):
        data.to(device)
        optimizer.zero_grad()

        n_corr = 0
        n_total = 0

        out = model(data.float()).squeeze()
        loss = loss_fn(out, label.long())
        loss.backward()
        optimizer.step()

        n_corr += (torch.argmax(out, dim = 1) == label.long()).sum().float().item()
        n_total += label.size()[0]

        acc = n_corr/n_total
        batch_acc_list.append(acc)
        batch_loss_list.append(loss)
        batch_count += 1
        #print(f"Batch Accuracy: {acc:.2f} | Batch loss: {loss:.2f}")
      epoch_loss = (sum(batch_loss_list)/batch_count)
      epoch_acc = (sum(batch_acc_list)/batch_count)
      print(f"Epoch: {epoch} | Loss: {epoch_loss} | Accuracy: {epoch_acc}")
      loss_list.append(epoch_loss)
    return loss_list

################################################################################
################# Training Step for VICReg ResNet x ResNet x Semi ##############
################################################################################
import os
import torch
import numpy as np
from torch import nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from IPython.display import clear_output

from thesis.loss import vicreg_loss_fn as vlf
from thesis.loss.similarity_loss import cosine_sim
from thesis.loss.similarity_loss import NCELoss

def train_vicreg_cnn_2(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  epochs: int,
  weight_vicreg: float = 1,
  sim_vicreg: float = 25,
  var_vicreg: float = 25,
  cov_vicreg: float = 1,
  decay_rate_vicreg: float = 0.01,
  decay_steps_vicreg: float = 100,
  weight_sim: float = 0.01,
  decay_rate_sim: float = 0.01,
  decay_steps_sim: float = -100,
  lr: float = 0.001,
  t: float = 1.,
  alpha: float = 0.5,
  alpha_prot: float = 0.3,
  epsilon: float = 0.05,
  instance_weight: float = 1,
  proto_weight: float = 5,
  cel_weight: float = 1,
  dist_weight: float = 500,
  num_classes: float = 3,
  sim_loss_fn: str = "cosine",
  lr_scheduler: str = "exp",
  gamma: float = 0.9,
  ssv_prob: float = 1,
  root_dir = None, **kwargs) -> torch.Tensor:

    """Training step for Self-Supervised Training model with VICReg and Supervised
    Training with Sim loss.

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
 
      model: torch.nn.Module: 
      dataloader: torch.utils.data.DataLoader: 
      epochs: int: 
      weight_vicreg: float:  (Default value = 1)
      sim_vicreg: float:  (Default value = 25)
      var_vicreg: float:  (Default value = 25)
      cov_vicreg: float:  (Default value = 1)
      decay_rate_vicreg: float:  (Default value = 0.01)
      decay_steps_vicreg: float:  (Default value = 100)
      weight_sim: float:  (Default value = 0.01)
      decay_rate_sim: float:  (Default value = 0.01)
      decay_steps_sim: float:  (Default value = -100)
      lr: float:  (Default value = 0.001)
      t: float:  (Default value = 1.)
      alpha: float:  (Default value = 0.5)
      alpha_prot: float:  (Default value = 0.3)
      epsilon: float:  (Default value = 0.05)
      instance_weight: float:  (Default value = 1)
      proto_weight: float:  (Default value = 5)
      cel_weight: float:  (Default value = 1)
      dist_weight: float:  (Default value = 500)
      num_classes: float:  (Default value = 3)
      sim_loss_fn: str:  (Default value = "cosine")
      lr_scheduler: str:  (Default value = "exp")
      gamma: float:  (Default value = 0.9)
      ssv_prob: float:  (Default value = 1)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.
      Gratefully adapted from: https://github.com/vturrisi/solo-learn

    """

    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model.train()
      model.to(device)

      # Initiate return variables
      loss_list, vicreg_loss_list, sim_loss_list, prototypes_list = [], [], [], []
      prototypes = None
      
      # Define optimizer and scheduler
      optimizer = torch.optim.Adam(model.parameters(), lr = lr)
      if lr_scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

      weight_vicreg_init = weight_vicreg
      weight_sim_init = weight_sim

      # Training loop
      for epoch in tqdm(range(epochs)):
          batch_count = batch_loss = vicreg_batch_loss = sim_batch_loss = epoch_loss = 0
          
          # VICReg Loss Decay
          if (bool(decay_rate_vicreg)) & (epoch <= abs(decay_steps_vicreg)):
              weight_vicreg = weight_vicreg_init*decay_rate_vicreg**(epoch/decay_steps_vicreg)
          
          # Sim Loss Decay
          if (bool(decay_rate_sim)) & (epoch <= abs(decay_steps_sim)):
              weight_sim = weight_sim_init*decay_rate_sim**(epoch/decay_steps_sim)

          # Batch loop
          for image_1, image_2, labels in tqdm(dataloader, leave = False):
              
              # Zero grads -> forward pass -> compute loss -> backprop
              optimizer.zero_grad()
              out = model(image_1.float(), image_2.float()).float().squeeze()
              feature_size = out.size()[1]

              labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1)
              
              # Calculate VICReg Loss function
              vicreg_loss = weight_vicreg*vlf.vicreg_loss_func(
                  out[:,:int(feature_size*0.5)],
                  out[:, int(feature_size*0.5):], sim_loss_weight = sim_vicreg,
                  var_loss_weight = var_vicreg, cov_loss_weight = cov_vicreg,
                  ).float()
              
              # Assign features for handling in following loss functions from model
              # outputs
              sim_features = torch.cat(
                              (
                                  out[:,:int(feature_size*0.5)],
                                  out[:, int(feature_size*0.5):]
                                ), dim = 0)
              
              # Implementation of cosine similarity loss
              if sim_loss_fn == "cosine":
                  sim_loss = weight_sim*cosine_sim(
                      sim_features, labels, t, alpha
                      ).float()
              
              #Implementation of NCELoss
              elif sim_loss_fn == "NCELoss":
                  sim_loss = weight_sim*NCELoss(sim_features, labels, t).float()

              #Implementation of prototypical loss
              elif sim_loss_fn == "proto":        
                  sim_loss, instance_loss, proto_loss, \
                  ce_loss, dist_loss, prototypes_updated = proto_sim(
                                reps = sim_features, labels = labels, 
                                prototypes = prototypes, 
                                t = t, alpha = alpha, alpha_prot = alpha_prot, 
                                instance_weight = instance_weight, 
                                proto_weight = proto_weight, dist_weight = dist_weight,
                                cel_weight = cel_weight, num_classes = num_classes,
                                epsilon = epsilon, epoch = epoch
                                )
                  
                  # Reassign prototypes
                  prototypes = prototypes_updated.detach()
                  sim_loss = weight_sim*sim_loss.float().detach()
              
              # Determine the probability with which supervised labels will be used
              semi_sup_ = int(np.random.choice(2, 1, p = [1- ssv_prob, ssv_prob]))
              loss = vicreg_loss + semi_sup_*sim_loss

              loss.backward()
              optimizer.step()

              # Output batch losses
              batch_count += 1
              batch_loss += loss.detach().cpu().numpy()
              vicreg_batch_loss += vicreg_loss.detach().cpu().numpy()
              sim_batch_loss += sim_loss.detach().cpu().numpy()
              print(f"Epoch: {epoch} | Batch_Loss: {loss.detach().cpu().numpy()}")
              
          clear_output()

          # Calculate and log epoch losses
          epoch_loss = batch_loss/batch_count
          vicreg_loss = vicreg_batch_loss/batch_count
          sim_loss = sim_batch_loss/batch_count

          if sim_loss_fn == "proto":
              wandb.log({
                          "loss":epoch_loss, 
                          "vicreg_loss": vicreg_loss, 
                          "sim_loss": sim_loss,
                          "sim_loss_norm": sim_loss/weight_sim,
                          "vicreg_loss_norm": vicreg_loss/weight_vicreg,
                          "weight_vicreg":weight_vicreg, 
                          "weight_sim":weight_sim,
                          "inst_loss":instance_loss,
                          "proto_loss":proto_loss,
                          "ce_loss":ce_loss,
                          "dist_loss":dist_loss,
                          "prototypes":prototypes_updated,
                        })
              prototypes_list.append(prototypes_updated.detach().cpu().numpy())

              
          elif sim_loss_fn == "cosine":
              wandb.log({
                          "loss":epoch_loss, 
                          "vicreg_loss": vicreg_loss, 
                          "sim_loss": sim_loss,
                          "sim_loss_norm": sim_loss/weight_sim,
                          "vicreg_loss_norm": vicreg_loss/weight_vicreg,
                          "weight_vicreg":weight_vicreg, 
                          "weight_sim":weight_sim,
                        })              
          loss_list.append(epoch_loss)
          vicreg_loss_list.append(vicreg_loss)
          sim_loss_list.append(sim_loss)

          print(f"Epoch: {epoch} | Epoch loss: {epoch_loss:.2f}")
            
          # Save model, in case a root_dir is given
          if (epoch > 5) & (root_dir is not None) & (~np.isnan(loss.detach().cpu().numpy())):
              if (loss < loss_list[-2]):
                  PATH = os.path.join(root_dir, f"{run_name}.pt")
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss,
                    }, PATH) 
        
            
      # Return loss logs and prototypes, in case it's given
      if sim_loss_fn == "proto":
          return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list
      else:
          return loss_list, vicreg_loss_list, sim_loss_list

    except KeyboardInterrupt:
        print("Execution interrupted by user")
        if sim_loss_fn == "proto":
            return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list
        else:
            return loss_list, vicreg_loss_list, sim_loss_list

################################################################################
####################### Training scheme VicReg CNNx2 and energy loss ###########
################################################################################


def train_vic_cnn_2_enloss(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  epochs: int,
  weight_vicreg: float = 1,
  sim_vicreg: float = 25,
  var_vicreg: float = 25,
  cov_vicreg: float = 1,
  decay_rate_vicreg: float = 0.01,
  decay_steps_vicreg: float = 100,
  weight_sim: float = 0.01,
  decay_rate_sim: float = 0.01,
  decay_steps_sim: float = -100,
  lr: float = 0.001,
  t: float = 1.,
  m: float = 0.5,
  alpha: float = 0.5,
  lr_scheduler: str = "exp",
  metric: str = "euclid",
  warm_up: int = 20,
  num_classes: int = 3,
  gamma: float = 0.9,
  root_dir: str = None,
  run_name: str = None, 
  **kwargs) -> torch.Tensor:

    """Training step for Self-Supervised Training model with VICReg and Sim loss

    Args:
      batch(Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
    [X] is a list of size num_crops containing batches of images.
      model: torch.nn.Module:
      dataloader: torch.utils.data.DataLoader:
      epochs: int:
      weight_vicreg: float:  (Default value = 1)
      sim_vicreg: float:  (Default value = 25)
      var_vicreg: float:  (Default value = 25)
      cov_vicreg: float:  (Default value = 1)
      decay_rate_vicreg: float:  (Default value = 0.01)
      decay_steps_vicreg: float:  (Default value = 100)
      weight_sim: float:  (Default value = 0.01)
      decay_rate_sim: float:  (Default value = 0.01)
      decay_steps_sim: float:  (Default value = -100)
      lr: float:  (Default value = 0.001)
      t: float:  (Default value = 1.)
      m: float:  (Default value = 0.5)
      alpha: float:  (Default value = 0.5)
      lr_scheduler: str:  (Default value = "exp")
      metric: str:  (Default value = "euclid")
      warm_up: int:  (Default value = 20)
      num_classes: int:  (Default value = 3)
      gamma: float:  (Default value = 0.9)
      root_dir: str:  (Default value = None)
      run_name: str:  (Default value = None)
      **kwargs: 
      model: torch.nn.Module: 
      dataloader: torch.utils.data.DataLoader: 
      epochs: int: 
      weight_vicreg: float:  (Default value = 1)
      sim_vicreg: float:  (Default value = 25)
      var_vicreg: float:  (Default value = 25)
      cov_vicreg: float:  (Default value = 1)
      decay_rate_vicreg: float:  (Default value = 0.01)
      decay_steps_vicreg: float:  (Default value = 100)
      weight_sim: float:  (Default value = 0.01)
      decay_rate_sim: float:  (Default value = 0.01)
      decay_steps_sim: float:  (Default value = -100)
      lr: float:  (Default value = 0.001)
      t: float:  (Default value = 1.)
      m: float:  (Default value = 0.5)
      alpha: float:  (Default value = 0.5)
      lr_scheduler: str:  (Default value = "exp")
      metric: str:  (Default value = "euclid")
      warm_up: int:  (Default value = 20)
      num_classes: int:  (Default value = 3)
      gamma: float:  (Default value = 0.9)
      root_dir: str:  (Default value = None)
      run_name: str:  (Default value = None)

    Returns:
      torch.Tensor: total loss composed of VICReg loss and classification loss.
      Gratefully adapted from: https://github.com/vturrisi/solo-learn

    """

    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model.train()
      model.to(device)
      prototypes = torch.Tensor().to(device)
      counts = torch.Tensor().to(device)

      # Initiate return variables
      loss_list, vicreg_loss_list, sim_loss_list, prototypes_list = [], [], [], []
      
      # Define optimizer and scheduler
      optimizer = torch.optim.Adam(model.parameters(), lr = lr)

      if lr_scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

      weight_vicreg_init = weight_vicreg
      weight_sim_init = weight_sim

      # Training loop
      for epoch in tqdm(range(epochs)):
          batch_count = batch_loss = vicreg_batch_loss = sim_batch_loss = epoch_loss = 0
          
          # VICReg Loss Decay
          if (bool(decay_rate_vicreg)) & (epoch <= abs(decay_steps_vicreg)):
              weight_vicreg = weight_vicreg_init*decay_rate_vicreg**(epoch/decay_steps_vicreg)
          
          # Sim Loss Decay
          if (bool(decay_rate_sim)) & (epoch <= abs(decay_steps_sim)):
              weight_sim = weight_sim_init*decay_rate_sim**(epoch/decay_steps_sim)

          # Batch loop
          for image_1, image_2, labels in tqdm(dataloader, leave = False):
              
              # Zero grads -> forward pass -> compute loss -> backprop
              optimizer.zero_grad()
              out = model(image_1.float(), image_2.float()).float().squeeze()
              feature_size = out.size()[1]
              labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1)
              
              # Calculate VICReg Loss function
              vicreg_loss = weight_vicreg*vlf.vicreg_loss_func(
                  out[:,:int(feature_size*0.5)],
                  out[:, int(feature_size*0.5):], sim_loss_weight = sim_vicreg,
                  var_loss_weight = var_vicreg, cov_loss_weight = cov_vicreg,
                  ).float()
              
              # Assign features for handling in following loss functions from model
              # outputs
              sim_features = torch.cat(
                              (
                                  out[:,:int(feature_size*0.5)],
                                  out[:, int(feature_size*0.5):]
                                ), dim = 0)
              
              #Implementation of prototypical loss
              sim_loss, prototypes, counts = energy_loss(
                  reps = sim_features, 
                  labels = labels,
                  prototypes = prototypes,
                  alpha = alpha,
                  metric = metric,
                  warm_up = warm_up,
                  epoch = epoch,
                  counts = counts,
                  t = t,
                  m = m,
                  )
              
              # Reassign prototypes
              #prototypes = prototypes.detach()
              sim_loss = weight_sim*sim_loss.float().detach()
              
              # Determine the probability with which supervised labels will be used
              loss = vicreg_loss + sim_loss
              loss.backward()
              optimizer.step()

              # Output batch losses
              batch_count += 1
              batch_loss += loss.detach().cpu().numpy()
              vicreg_batch_loss += vicreg_loss.detach().cpu().numpy()
              sim_batch_loss += sim_loss.detach().cpu().numpy()
              print(f"Epoch: {epoch} | Batch_Loss: {loss.detach().cpu().numpy()}")
              
          clear_output()

          # Calculate and log epoch losses
          epoch_loss = batch_loss/batch_count
          vicreg_loss = vicreg_batch_loss/batch_count
          sim_loss = sim_batch_loss/batch_count

          wandb.log({
                      "loss":epoch_loss, 
                      "vicreg_loss": vicreg_loss, 
                      "sim_loss": sim_loss,
                      "sim_loss_norm": sim_loss/weight_sim,
                      "vicreg_loss_norm": vicreg_loss/weight_vicreg,
                      "weight_vicreg":weight_vicreg, 
                      "weight_sim":weight_sim,
                      "prototypes":prototypes.detach(),
                    })
          
          prototypes_list.append(prototypes.detach().cpu().numpy())
              
          loss_list.append(epoch_loss)
          vicreg_loss_list.append(vicreg_loss)
          sim_loss_list.append(sim_loss)

          print(f"Epoch: {epoch} | Epoch loss: {epoch_loss:.2f}")

          # Save model, in case a root_dir is given
          if (root_dir != None) & (loss < loss_list[-1]):
            PATH = os.path.join(root_dir, f"{run_name}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)          
            
      # Return loss logs and prototypes, in case it's given
      return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list

    except KeyboardInterrupt:
        print("Execution interrupted by user")
        return loss_list, vicreg_loss_list, sim_loss_list, prototypes_list

def resnet18_training(
    model: torch.nn.Module = None, 
    dataloader_list: list = dataloader_list,
    dataloader_test_list: list = dataloader_test_list,
    num_classes: int = 2,
    epoch_list: list = [25, 50, 75, 100],
    loss_fn: str = "energy",
    lr_scheduler: str = "exp",
    lr: float = 0.001,
    gamma: float = 0.9,
    path: str = "/content/drive/MyDrive/MT Gabriel/model_runs/",
    run_name: str = None,
    **kwargs,
    ) -> [list, list]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    if lr_scheduler == "exp":
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

    train_acc_list = []
    loss_list = []

    # Initialize counters and dataloaders
    epochs_limit = 0
    epochs_limit_counter = 0
    epochs_limit = epoch_list[epochs_limit_counter]
    dataloader = dataloader_list[epochs_limit_counter]
    dataloader_test = dataloader_test_list[epochs_limit_counter]
    epochs = epoch_list[-1]

    for epoch in tqdm(range(epochs)):
        corrects = torch.Tensor([0]).to(device)
        total = torch.Tensor([0]).to(device)

        # Training loop
        for data_1, data_2, label in tqdm(dataloader, leave = False):
            optimizer.zero_grad()
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            labels = label.to(device)
            outs_1 = model(data_1)
            outs_2 = model(data_2)
            outs = torch.cat((outs_1, outs_2), 0)

            labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1).squeeze()

            # Criterion to use
            if loss_fn == "energy":
                loss = energy_logit_loss(outs, labels)
                _, preds = torch.min(outs, 1)
            
            elif loss_fn == "entropy":
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(outs, labels)
                _, preds = torch.max(outs, 1)                

            loss.backward()
            optimizer.step()

            # Calculate Acc
            corrects += torch.sum(preds == labels)
            total += torch.numel(labels)
        
        # Calculate epoch accuracy
        epoch_acc = (corrects/total).detach().cpu().numpy()
        wandb.log({"train_acc":epoch_acc, "loss":loss})

        train_acc_list.append(epoch_acc)
        loss_list.append(loss)
        print(f"Epoch: {epoch} | Epoch_acc: {epoch_acc} | Loss: {loss}")

        # Update counters, write Classification report to file, re_allocate
        # dataloaders
        if epoch == epochs_limit - 1:
            test_model = copy.deepcopy(model)
            resnet_eval(test_model, 
                        dataloader_test, 
                        num_classes = epochs_limit_counter + 2,
                        loss_fn = loss_fn,
                        path = path + run_name + ".txt"
                        )
            
            epochs_limit_counter += 1
            #num_classes += 1
            epochs_limit = epoch_list[epochs_limit_counter]
            dataloader = dataloader_list[epochs_limit_counter]
            dataloader_test = dataloader_test_list[epochs_limit_counter]
            print(f"Using dataloader no. {epochs_limit_counter}")

    return loss_list, train_acc_list

################################################################################
#ProReC training ###############################################################
################################################################################
def resnet18_training(
    model: torch.nn.Module = None,
    dataloader_list: list = dataloader_list,
    dataloader_test_list: list = dataloader_test_list,
    total_classes: int = 5,
    cls_per_run: int = 2,
    epochs: int = 100,
    epoch_list: list = [25, 50, 75, 100],
    sim_loss_weight: float = 25.,
    cov_loss_weight: float = 25.,
    var_loss_weight: float = 1.,
    dist_loss_weight: float = 1.,
    proto_reg_loss_weight: float = 1.,
    loss_fn: str = "proto_vicreg",
    proto_feature_size: int = 256,
    proj_hidden_size: int = 512,
    proj_layers: int = 1,
    max_storage_per_cls: int = 64,
    new_samples_batch: int = 24,
    store = "random",
    lr_scheduler: str = "exp",
    lr: float = 0.00001,
    lr_proj: float = 0.001,
    gamma: float = 0.9,
    path: str = "/content/drive/MyDrive/MT Gabriel/model_runs/",
    run_name: str = None,
    **kwargs,
    ) -> [list, list]:
    torch.autograd.set_detect_anomaly(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    num_features = model.fc.in_features

    layers_dict = OrderedDict()
    layers = []
    layers.append(nn.Linear(num_features, proj_hidden_size))
    layers.append(nn.ReLU())
    while len(layers) < proj_layers*2:
        layers.append(nn.Linear(proj_hidden_size, proj_hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(proj_hidden_size, proj_hidden_size))
    # layers.append(nn.Linear(proj_hidden_size, proto_feature_size))

    for k, v in enumerate(layers):
        layers_dict[str(k)] = v
        
    model.fc = nn.Sequential(layers_dict)
    model.to(device)

    proj_params = []
    base_params = []
    for i,j in model.named_parameters():
        if "fc" in i:
            proj_params.append(j)
        else:
            base_params.append(j)

    optimizer = torch.optim.Adam([
                                  {"params":base_params},
                                  {"params":proj_params, "lr" : lr_proj},
    ], lr = lr)

    # if lr_scheduler == "exp":
    #   scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

    train_acc_list = []
    loss_list = []
    forgetting_per_cls_list = []
    forgetting_m_list = []

    # Initialize counters and dataloaders
    epochs_limit = 0
    epochs_limit_counter = 0

    num_classes = cls_per_run +  epochs_limit_counter
    #num_classes = num_classes

    epochs_limit = epoch_list[epochs_limit_counter]
    dataloader = dataloader_list[epochs_limit_counter]
    dataloader_test = dataloader_test_list[epochs_limit_counter]
    protos = torch.Tensor().to(device)
    protos_activation = torch.Tensor().to(device)

    data_storage = torch.Tensor().to(device)
    label_storage = torch.Tensor().to(device)
    outs_storage = torch.Tensor().to(device)

    test_acc_lists = []

    try:
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        for epoch in tqdm(range(epochs)):
            corrects = torch.Tensor([0]).to(device)
            total = torch.Tensor([0]).to(device)
            epoch_loss_list = []
            epoch_uncertainty_loss_list = []

            model.avgpool.register_forward_hook(get_activation("avgpool"))

            # Training loop
            for data_1, data_2, label in tqdm(dataloader, leave = False):
                optimizer.zero_grad()
                data_2 = data_2.to(device)
                labels = label.to(device)

                labels = labels.view(labels.size(dim = 0), 1).repeat(2, 1).squeeze()
                data = torch.cat((data_1, data_2), dim = 0)
                outs = model(data)
                # outs_1 = model(data_1)
                # outs_2 = model(data_2)
                # outs = torch.cat((outs_1, outs_2), 0)
                
                # if data_storage.size()[0] > 10:
                #     outs_storage = model(data_storage)

                if loss_fn == "proto_vicreg":
                    # print(f"Protos@loss:{protos}")
                    act = activation["avgpool"]
                    loss, protos_out, protos_act = proto_vicreg_loss_func(
                        model,
                        outs, labels, 
                        num_classes = num_classes,  
                        protos = protos,
                        protos_act = protos_activation,
                        act = act.squeeze(),
                        sim_loss_weight = sim_loss_weight,
                        var_loss_weight = var_loss_weight,
                        cov_loss_weight = cov_loss_weight,
                        dist_loss_weight = dist_loss_weight,
                        proto_reg_loss_weight = proto_reg_loss_weight,
                        outs_storage = data_storage,
                        label_storage = label_storage,
                        )

                    protos = protos_out.detach().clone()
                    protos_activation = protos_act.detach().clone()
                    
                    preds, labels_total = get_prediction(
                        outs, labels, 
                        outs_storage = torch.Tensor().to(device), 
                        label_storage = torch.Tensor().to(device), 
                        protos = protos.detach(),
                        )

                # Calculate Acc
                corrects += torch.sum(preds == labels_total)
                total += torch.numel(labels)
                print(f"Loss: {loss}")

                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                epoch_loss_list.append(loss.detach().cpu().numpy())

                if max_storage_per_cls > 0:
                    if store == "random":
                        # Store samples from intermediate layer
                        for i in np.unique(labels.detach().cpu().numpy()):
                            labels_half = labels[:int(labels.size()[0]/2)].detach()
                            mask = i == labels_half.squeeze()
                            # print(f"Storage Samples in cls {i}: {(i == label_storage.squeeze()).sum()}")
                            # Check if > max_storage_per_cls samples per class -> drop the oldest
                            new_samples = new_samples_batch
                            keep_from_old = max_storage_per_cls - 2*new_samples
                            if (i == label_storage.squeeze()).sum() >= max_storage_per_cls - 2*new_samples:
                                data_storage = torch.cat(
                                    (
                                      data_storage[i != label_storage.squeeze()],
                                      data_storage[i == label_storage.squeeze()][-keep_from_old:]
                                    )
                                )
                                label_storage = torch.cat(
                                    (
                                    label_storage[i != label_storage.squeeze()],
                                    label_storage[i == label_storage.squeeze()][-keep_from_old:]
                                    )
                                )
                            act = act.squeeze()
                            act_1 = act[:int(labels.size()[0]/2)]
                            act_2 = act[int(labels.size()[0]/2):]

                            data_storage = torch.cat((data_storage, act_1[mask][:new_samples].detach()))
                            data_storage = torch.cat((data_storage, act_2[mask][:new_samples].detach()))
                            label_storage = torch.cat((label_storage, labels_half[mask][:new_samples].repeat(2)))

                    elif store == "nearest":
                        for i in np.unique(labels.detach().cpu().numpy()):
                            labels_half = labels[:int(labels.size()[0]/2)].detach()
                            mask = i == labels_half.squeeze()
                            # print(f"Storage Samples in cls {i}: {(i == label_storage.squeeze()).sum()}")
                            # Check if > max_storage_per_cls samples per class -> drop the oldest
                            num_labels = np.unique(labels.detach().cpu().numpy())
                            new_samples = int(new_samples_batch/len(num_labels))
                            keep_from_old = max_storage_per_cls - 2*new_samples_batch
                            if (i == label_storage.squeeze()).sum() >= max_storage_per_cls - 2*new_samples_batch:
                                data_storage = torch.cat(
                                    (
                                      data_storage[i != label_storage.squeeze()],
                                      data_storage[i == label_storage.squeeze()][-keep_from_old:]
                                    )
                                )
                                label_storage = torch.cat(
                                    (
                                    label_storage[i != label_storage.squeeze()],
                                    label_storage[i == label_storage.squeeze()][-keep_from_old:]
                                    )
                                )

                            act = act.squeeze()
                            print(act)

                            act_1 = act[:int(labels_half.size()[0])]
                            act_2 = act[int(labels_half.size()[0]):]

                            act_1_dist = torch.mean(F.mse_loss(act_1[mask], 
                                                 protos_activation[i].repeat(act_1[mask].size()[0], 1),
                                                 reduction = "none"
                                                 ), dim = 1)
                            act_2_dist = torch.mean(F.mse_loss(act_2[mask], 
                                                 protos_activation[i].repeat(act_2[mask].size()[0], 1),
                                                 reduction = "none"
                                                 ), dim = 1)
                            k_1 = np.min([new_samples, torch.numel(act_1_dist)])
                            k_2 = np.min([new_samples, torch.numel(act_2_dist)])
                            _, idx_1 = torch.topk(act_1_dist, k = k_1, largest = False)
                            _, idx_2 = torch.topk(act_2_dist, k = k_2, largest = False)

                            data_storage = torch.cat((data_storage, act_1[mask][idx_1].detach()))
                            data_storage = torch.cat((data_storage, act_2[mask][idx_2].detach()))
                            label_storage = torch.cat((label_storage, labels_half[mask][idx_1]))
                            label_storage = torch.cat((label_storage, labels_half[mask][idx_2]))

            # Calculate epoch accuracy
            epoch_acc = (corrects/total).detach().cpu().numpy()

            train_acc_list.append(epoch_acc)
            loss_list.append(np.mean(epoch_loss_list))
            print(f"Epoch: {epoch} | Epoch_acc: {epoch_acc} | Loss: {np.mean(epoch_loss_list)}")

            test_model = copy.deepcopy(model)
            test_acc, test_acc_list = resnet_eval(
                test_model, 
                dataloader_test, 
                num_classes = num_classes,
                protos = protos,
                loss_fn = loss_fn,
                path = path + run_name + ".txt"
                )
            test_acc_lists.append(test_acc_list)
            # Calculate Forgetting
            forgetting_m, forgetting_per_cls = forgetting_metric(
                test_acc_lists, num_classes, total_classes, cls_per_run,
                )
            forgetting_m_list.append(forgetting_m)
            forgetting_per_cls_list.append(forgetting_per_cls)

            overall_metric = 0.5*(1-forgetting_m)+0.5*test_acc
            if epoch == epoch_list[-1] - 1:
                labels_dict = {
                    0.0: "regular", 
                    1.0: "fold", 
                    2.0: "gap", 
                    3.0: "und", 
                    4.0: "regular_ncf",
                    5.0: "fold_ncf",
                    6.0: "gap_ncf",
                    7.0: "und_ncf",
                    8.0: "p_reg",
                    9.0: "p_fold",
                    10.0: "p_gap",
                    11.0: "p_und",
                    12.0:"p_reg_ncf",
                    13.0:"p_fold_ncf",
                    14.0: "p_gap_ncf",
                    15.0:"p_und_ncf"
                    }

                visualize_embeddings(
                    test_model, 
                    dataloader_test, 
                    prototypes = protos,
                    outs_storage = torch.Tensor().to(device),
                    label_storage = torch.Tensor().to(device),
                    projected = True,
                    labels_dict = labels_dict,
                    )
                
            wandb.log({
                "train_acc":epoch_acc, 
                "test_acc":test_acc, 
                "loss":np.mean(epoch_loss_list),
                "forgetting":forgetting_m,
                "test_acc_list":test_acc_list,
                "overall_metric":overall_metric,
                })
            torch.cuda.empty_cache()

            ########################################################################
            # For continual learning setting #######################################
            ########################################################################
            # Update counters, write Classification report to file, re_allocate
            # dataloaders

            if (epoch == epochs_limit - 1) & (epochs_limit != epoch_list[-1]):
                test_model = copy.deepcopy(model)

                # Update counters and dataloaders
                epochs_limit_counter += 1
                num_classes = cls_per_run + epochs_limit_counter
                epochs_limit = epoch_list[epochs_limit_counter]
                dataloader = dataloader_list[epochs_limit_counter]
                dataloader_test = dataloader_test_list[epochs_limit_counter]

                print(f"Using dataloader no. {epochs_limit_counter}")
        
        torch.save(test_acc_lists, path + run_name + "_test_acc.txt")
        torch.save(forgetting_per_cls_list, path + run_name + "_forgetting_list.txt")

        run.log_artifact(path + run_name + "_test_acc.txt", type = "test_acc_lists")
        run.log_artifact(path + run_name + "_test_acc.txt", type = "forgetting_list")
        return loss_list, test_acc_lists, forgetting_m_list, forgetting_per_cls_list, model, protos
        
    except KeyboardInterrupt:
        return loss_list, test_acc_lists, forgetting_m_list, forgetting_per_cls_list, model, protos