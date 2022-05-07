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