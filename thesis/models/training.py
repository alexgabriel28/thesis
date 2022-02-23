#############################################################
################# Training Step for GNN #####################
#############################################################
import torch.optim as optim
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Train the given model on the given dataset for num_epochs
def train_gat(model, train_loader, test_loader, num_epochs, edge_attr = True):

    # Set up the loss and the optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_acc_ls = []
    test_acc_ls = []
    loss_ls = []

    # A utility function to compute the accuracy
    def get_train_acc(model, loader):
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

    # Set up the loss and the optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_acc_ls = []
    test_acc_ls = []
    loss_ls = []

    # A utility function to compute the accuracy
    def get_train_acc(model, loader):
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
        batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
            [X] is a list of size num_crops containing batches of images.
        batch_idx (int): index of the batch.
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

            vicreg_loss = vicreg_loss_fn(
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