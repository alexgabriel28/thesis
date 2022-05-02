import torch
def proto_sim(reps: torch.Tensor,
                labels: torch.Tensor,
                prototypes: torch.Tensor,
                t: float,
                alpha: float,
                alpha_prot: float,
                instance_weight: float,
                proto_weight: float,
                cel_weight: float,
                dist_weight: float,
                num_classes: int,
                epsilon: float,
                epoch: int,
                ) -> torch.Tensor:
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert (alpha >= 0) & (alpha <= 1), "alpha must be in [0,1]"
    assert (alpha_prot >= 0) & (alpha_prot <= 1), "alpha_prot must be in [0,1]"

    if epoch == 0:
        prototypes = torch.zeros(
            len(torch.unique(labels)), reps.size()[1]
            ).to(device)
        for i in torch.unique(labels):
            trues = labels == i
            trues = trues.view(-1, 1)
            prototypes[i] = (trues.T.float() @ reps)/torch.sum(trues)

    # For numerical stability of exp function
    eps = torch.Tensor([1e-08]).to(device)
    Eps = torch.Tensor([1.797693134862315e+308]).to(device)

    reps = reps/torch.sqrt(torch.sum(reps.pow(2), dim = 1, keepdim = True))
    prototypes = prototypes/torch.sqrt(torch.sum(prototypes.pow(2), dim = 1, keepdim = True))

    # Calculate l2-norms of all vector combinations 
    v_1, v_2 = torch.sum(torch.square(reps), dim = 1).view(reps.size(0), 1), \
        torch.sum(torch.square(reps), dim = 1).view(1, reps.size(0))

    v_p1, v_p2 = torch.sum(torch.square(prototypes), dim = 1).view(prototypes.size(0), 1), \
        torch.sum(torch.square(prototypes), dim = 1).view(1, prototypes.size(0))

    # Note: matmul of v_1 with v_p2 to get 100 x 3 norm_matrix_proto
    norm_matrix = torch.matmul(torch.sqrt(v_1), torch.sqrt(v_2))
    norm_matrix_proto = torch.matmul(torch.sqrt(v_1), torch.sqrt(v_p2))

    # Calculate vector (cosine) similarities and normalize by l2 norms of vectors
    sim = torch.matmul(reps, reps.T)/(torch.max(eps, norm_matrix)*t)
    #sim = sim/torch.sum(sim, dim = 1, keepdim = True)

    sim_p = torch.matmul(reps, prototypes.T)/(torch.max(eps, norm_matrix_proto)*t)
    sim_p = torch.clamp(torch.exp(sim_p + 0), min = eps, max = Eps)

    #sim_p = sim_p/torch.sum(sim_p, dim = 1, keepdim = True)

    # Concat output from head_1 and head_2 and choose best prototype by "ensemble" voting
    ensemble = torch.cat((
        sim_p[:int(0.5*sim_p.size()[0]), :], 
        sim_p[int(0.5*sim_p.size()[0]):, :]), 
        dim = 1
        )

    # Find the prototype p_c with the shortest distance to two corresponding image patches
    proto_max = torch.argmax(ensemble, dim = 1)
    proto_max_tot = torch.LongTensor([entry if entry < sim_p.size()[1] else entry - sim_p.size()[1] for entry in proto_max]).to(device)
    proto_c = torch.cat((proto_max_tot, proto_max_tot))

    # Calculate mean of the sims of instances of the same (positive)
    # prototypes of one class: p_c_+

    # If calculated from the predictions
    #proto_labels = torch.nn.functional.one_hot(proto_c, num_classes = num_classes)

    # If calculated from the labels
    proto_labels = torch.nn.functional.one_hot(labels.squeeze(), num_classes = num_classes)

    proto_pos_loss = alpha_prot*(torch.mean(proto_labels.float() @ sim_p.T))
    proto_neg_loss = (1 - alpha_prot)*torch.mean((abs(proto_labels.float() - 1) @ sim_p.T))

    # Calculate the CrossEntropyLoss -> correct classification of instance i_c to p_c
    cel = torch.nn.CrossEntropyLoss()
    ce_loss = cel(sim_p, labels.squeeze())

    # Delete "self-loops" from similarity matrix by subtracting diagonal values
    sim = sim - torch.diag(torch.diagonal(sim))
    # Add zero for stability and clamp to float32 values
    sim = torch.clamp(torch.exp(sim + 0), min = eps, max = Eps)

    # Finds which instances are of the same class
    # If cls1 == cls2 -> label_1 - label_2 == 0
    # If cls 1 != cls2 -> abs(label_1 - label_2) >= 0
    proto_class = proto_c.view(-1, 1)
    pos_mask = (~torch.abs(labels.T - labels).bool()).float()
    neg_mask = (torch.abs(labels.T - labels).bool()).float()

    # Average positive and negative similarities for a batch and weight by alpha
    pos_loss = torch.mean(alpha*(pos_mask*sim))
    neg_loss = torch.mean((1 - alpha)*(neg_mask*sim))

    # Update Prototypes
    prototypes_updated = prototypes.clone().detach()

    for i in range(prototypes.size()[0]):
        label_i = labels == i
        prototype_new = (epsilon*prototypes_updated[i] + (1-epsilon)*(torch.mean(label_i.float().T @ reps)/torch.sum(label_i)))
        prototypes_updated[i] = prototype_new/torch.linalg.vector_norm(prototype_new)

    # Calculate the distances of the prototypes -> L2-Norm normalization
    proto_dist = torch.Tensor([0]).to(device)
    for i, prototype in enumerate(prototypes_updated):
        proto_dist += torch.sqrt(torch.sum(torch.square(prototypes_updated - prototype)))
    proto_dist = proto_dist/prototypes_updated.size()[0]
    
    # Sum up and weigh the different losses
    instance_loss = instance_weight*(neg_loss - pos_loss)
    proto_loss = proto_weight*(proto_neg_loss - proto_pos_loss)
    ce_l = cel_weight*ce_loss
    dist_l = dist_weight*proto_dist
    loss =  instance_loss + proto_loss + ce_l - dist_l.squeeze()

    # Return overall loss
    return loss.to(reps.device), instance_loss, proto_loss, ce_l, dist_l, prototypes_updated
    #return loss.to(reps.device), instance_loss, proto_loss, ce_l, prototypes_updated