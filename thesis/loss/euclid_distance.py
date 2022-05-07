import torch.nn.functional as F
def euclid_distance(
    reps: torch.Tensor = None,
    labels: torch.Tensor = None,
    fraction: float = 1.,
    num_classes: int = 5,
    return_proto: bool = True,
    protos: torch.Tensor = None,
    cls_counter: int = 0,    
) -> torch.Tensor:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For numerical stability of exp function
    eps = torch.Tensor([1e-08]).to(device)
    Eps = torch.Tensor([1.797693134862315e+308]).to(device)
    
    # Use only the fraction of data, specified in fraction -> semi-supervised
    if fraction < 1.:
        erase_v = (torch.rand(size=(reps.size()[0], 1)) < fraction).to(device).float()
        reps = erase_v * reps

    # Use this to calculate the loss, in case only one randomly sampled class
    # acts as neg representations
    classes = np.unique(labels.detach().cpu().numpy())
    proto_mses = torch.Tensor([0]).to(device)
    pos_mses = torch.Tensor([0]).to(device)
    neg_mses = torch.Tensor([0]).to(device)
    neg_mses = torch.Tensor([0]).to(device)

    # Iterate over all classes in classes and specify as pos_class
    for i, pos_cls in enumerate(classes):
        reps_pos = reps[labels.squeeze() == pos_cls]
        pos_mse = F.mse_loss(reps_pos, reps_pos[torch.randperm(reps_pos.size()[0])]).to(device)
        pos_mses = torch.cat((pos_mses, pos_mse.unsqueeze(dim = 0)))
        
        if cls_counter == 0:
            reps_neg = reps[labels.squeeze() != pos_cls]
            if reps_neg.size()[0] != 0:
                neg_mse = F.mse_loss(
                    reps_pos, 
                    reps_neg[torch.randint(
                        low = 0, 
                        high = reps_neg.size()[0], 
                        size = (reps_pos.size()[0],)
                        )
                    ])
                neg_mses = torch.cat((neg_mses, torch.mean(neg_mse, dim = 0).unsqueeze(dim = 0)))
            else:
                pass

        elif cls_counter > 0:
            protos = protos[torch.randperm(protos.size()[0])]
            for proto in protos[:pos_cls]:
                proto_mses = torch.cat((proto_mses, F.mse_loss(reps_pos, proto.repeat(reps_pos.size()[0], 1)).unsqueeze(dim = 0)))
            neg_mses = torch.cat((neg_mses, torch.mean(proto_mses, dim = 0, keepdim = True)))

    pos_loss = pos_mses.mean()
    neg_loss = neg_mses.mean()

    # Update protos every batch
    for i, proto in enumerate(protos):
        mask = labels.squeeze() == i
        if mask.sum() > 0:
            protos[i] = reps[labels.squeeze() == i].mean(0, keepdim = True).to(device)

    if return_proto == True:
        batch_mean_cls = reps[labels.squeeze() == cls_counter].mean(0, keepdim = True).to(device)
        protos = torch.cat((protos, batch_mean_cls)).to(device)
    print(f"pos_loss: {pos_loss:.2f}")
    print(f"neg_loss: {neg_loss:.2f}")
    return pos_loss - neg_loss, protos
    # else:
    #     return pos_loss - neg_loss