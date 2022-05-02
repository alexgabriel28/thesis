import torch

################################################################################
# Energy-based contrastive loss function: exp(E(x, y+)) - exp(E(x, y-))
# E(x) = cos(x_1, x_2)/(||x_1||*||x_2||)
################################################################################
def cosine_sim(
                reps: torch.Tensor = None,
                labels: torch.Tensor = None,
                t: float = 0.3,
                alpha: float = 0.3,
                fraction: float = 1.,
                num_classes: int = 5,
                neg_agg_choice: str = "proto",
                neg_selection: bool = True,
                return_proto: bool = True,
                protos: torch.Tensor = None,
                cls_counter: int = 0,
                ) -> torch.Tensor:
                
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert (alpha >= 0) & (alpha <=1), "Alpha must be in [0,1]"
    
    # For numerical stability of exp function
    eps = torch.Tensor([1e-08]).to(device)
    Eps = torch.Tensor([1.797693134862315e+308]).to(device)

    # Calculate l2-norms of all vector combinations 
    v_1, v_2 = torch.sum(torch.square(reps), dim = 1).view(reps.size(0), 1), \
        torch.sum(torch.square(reps), dim = 1).view(1, reps.size(0))

    norm_matrix = norm_matrix_pn = torch.matmul(
                    reps.norm(dim = -1, keepdim = True).view(reps.size(0), 1), 
                    reps.norm(dim = -1, keepdim = True).view(1, reps.size(0))
                    )
    # Use only the fraction of data, specified in fraction -> semi-supervised
    if fraction < 1.:
        erase_v = (torch.rand(size=(reps.size()[0], 1)) < fraction).to(device).float()
        reps = erase_v * reps

    # Calculate vector (cosine) similarities and normalize by l2 norms of vectors
    sim = torch.matmul(reps, reps.T)/(torch.max(eps, norm_matrix)*t)
    # Delete "self-loops" from similarity matrix by subtracting diagonal values
    sim = sim - torch.diag(torch.diagonal(sim))

    # Add zero for stability and clamp to float32 values
    sim = torch.clamp(torch.exp(sim + 0), min = eps, max = Eps)

    # Finds which instances are of the same class
    # If cls1 == cls2 -> label_1 - label_2 == 0
    # If cls 1 != cls2 -> abs(label_1 - label_2) >= 0
    pos_mask = (~torch.abs(labels.T - labels).bool()).float()

    # Average positive and negative similarities for a batch and weight by alpha
    pos_loss = torch.mean(alpha*(pos_mask*sim))

    ############################################################################
    #pos_loss = torch.mean(alpha*(pos_mask*2*(1-sim)))
    ############################################################################

    # Use this to calculate the loss, in case only one randomly sampled class
    # acts as neg representations
    if neg_selection == True:
        classes = np.unique(labels.detach().cpu().numpy())
        neg_loss = torch.Tensor([0]).to(device)
        neg_proto_loss = torch.Tensor([0]).to(device)

        # Iterate over all classes in classes and specify as pos_class
        for i, pos_cls in enumerate(classes):
            neg_class = np.random.choice(classes[classes != pos_cls])
            labels_neg = labels[labels == neg_class]
            neg_reps = reps[labels.squeeze() == neg_class]
            #print(f"neg_reps: {neg_reps}, nr.size: {neg_reps.size()}")
            pos_reps = reps[labels.squeeze() == pos_cls]

            # Calculate neg part of loss from proto from the one sampled class
            if neg_agg_choice == "proto":
                neg_proto = neg_reps.mean(0, keepdim = True)

                reps.norm(dim = -1, keepdim = True).view(reps.size(0), 1),

                proto_norm = neg_proto.norm(dim=-1, keepdim = True).view(1, neg_proto.size(0))
                reps_norm = pos_reps.norm(dim=-1, keepdim = True).view(pos_reps.size(0), 1)
                norm_matrix = torch.matmul(reps_norm, proto_norm)
                sim_neg = (pos_reps @ neg_proto.T)/(torch.max(eps, norm_matrix*t))
                sim_neg = torch.clamp(torch.exp(sim_neg + 0), min = eps, max = Eps)
                neg_loss += torch.mean((1-alpha)*sim_neg)

            # Calculate neg part of loss with all instances of the randomly
            # selected negative class for each class in classes
            elif neg_agg_choice == "single":
                # Calc norms of vectors
                norm_matrix_pn = torch.matmul(
                    pos_reps.norm(dim = -1, keepdim = True).view(pos_reps.size(0), 1), 
                    neg_reps.norm(dim = -1, keepdim = True).view(1, neg_reps.size(0))
                    )
                # Calc similarities 
                sim_pn = torch.maximum(torch.Tensor([0]).to(device), torch.matmul(pos_reps, neg_reps.T)/(torch.max(eps, norm_matrix_pn)*t) - 0.5)
                #print(f"Cosine sim: {sim_pn}")
                #print(sim_pn.size())
                sim_pn = torch.clamp(torch.exp(sim_pn + 0), min = eps, max = Eps)
                #print(f"After exp: {sim_pn}")

                ################################################################
                #sim_pn = -2*(1-sim_pn)
                ################################################################

                neg_loss += (1-alpha)*torch.mean(sim_pn)

                if cls_counter > 0:
                    neg_proto = protos
                    proto_norm = neg_proto.norm(dim=-1, keepdim = True).view(1, neg_proto.size(0))
                    reps_norm = pos_reps.norm(dim=-1, keepdim = True).view(pos_reps.size(0), 1)
                    norm_matrix = torch.matmul(reps_norm, proto_norm)
                    sim_neg = torch.maximum(torch.Tensor([0]).to(device), (pos_reps @ neg_proto.T)/(torch.max(eps, norm_matrix*t)) - 0.5)
                    sim_pn = torch.clamp(torch.exp(sim_pn + 0), min = eps, max = Eps)
                    ############################################################
                    #sim_neg = -2*(1-sim_neg)
                    ############################################################

                    neg_proto_loss += torch.mean((1-alpha)*sim_neg)
                else:
                    neg_proto_loss = torch.Tensor([0])
        
        # Use to return 0-dim tensor, fast work-around
        neg_loss = torch.sum(neg_loss)
        proto_loss = torch.sum(neg_proto_loss)

    else:
        neg_mask = (torch.abs(labels.T - labels).bool()).float()
        neg_loss = torch.mean((1 - alpha)*(neg_mask*sim))

    # Running batch mean
    batch_mean_cls = reps[labels.squeeze() == cls_counter].mean(0, keepdim = True).to(device)

    # Calc final sim loss
    loss =  (neg_loss - pos_loss)
    ############################################################################
    #loss = (pos_loss + neg_loss)
    ############################################################################
    if return_proto == True:
        protos = torch.cat((protos, batch_mean_cls)).to(device)
        return loss.to(reps.device), proto_loss, protos
    else:
        # Return overall loss
        return loss.to(reps.device), proto_loss


def cosine_sim(
                reps: torch.Tensor,
                labels: torch.Tensor,
                t: float,
                alpha: float,
                ) -> torch.Tensor:
                
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert (alpha >= 0) & (alpha <=1), "Alpha must be in [0,1]"
    
    # For numerical stability of exp function
    eps = torch.Tensor([1e-08]).to(device)
    Eps = torch.Tensor([1.797693134862315e+308]).to(device)

    # Calculate l2-norms of all vector combinations 
    v_1, v_2 = torch.sum(torch.square(reps), dim = 1).view(reps.size(0), 1), \
        torch.sum(torch.square(reps), dim = 1).view(1, reps.size(0))
    norm_matrix = torch.matmul(torch.sqrt(v_1), torch.sqrt(v_2))

    # Calculate vector (cosine) similarities and normalize by l2 norms of vectors
    sim = torch.matmul(reps, reps.T)/(torch.max(eps, norm_matrix)*t)
    # Delete "self-loops" from similarity matrix by subtracting diagonal values
    sim = sim - torch.diag(torch.diagonal(sim))
    # Add zero for stability and clamp to float32 values
    sim = torch.clamp(torch.exp(sim + 0), min = eps, max = Eps)

    # Finds which instances are of the same class
    # If cls1 == cls2 -> label_1 - label_2 == 0
    # If cls 1 != cls2 -> abs(label_1 - label_2) >= 0
    pos_mask = (~torch.abs(labels.T - labels).bool()).float()
    neg_mask = (torch.abs(labels.T - labels).bool()).float()

    # Average positive and negative similarities for a batch and weight by alpha
    pos_loss = torch.mean(alpha*(pos_mask*sim))
    neg_loss = torch.mean((1 - alpha)*(neg_mask*sim))
    loss =  (neg_loss - pos_loss)
    
    # Return overall loss
    return loss.to(reps.device)
################################################################################
# Noise Contrastive Estimation
################################################################################
def NCELoss(
                reps: torch.Tensor,
                labels: torch.Tensor,
                t: float,
                ) -> torch.Tensor:
                
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    v_1, v_2 = torch.sum(torch.square(reps), dim = 1).view(reps.size(0), 1), \
        torch.sum(torch.square(reps), dim = 1).view(1, reps.size(0))

    norm_matrix = torch.matmul(torch.sqrt(v_1), torch.sqrt(v_2))

    sim = torch.exp(torch.mm(reps, reps.T)/(norm_matrix * t))
    pos_mask = (~torch.abs(labels.T - labels).bool()).float()
    neg_mask = (torch.abs(labels.T - labels).bool()).float()
    pos_sim = torch.sum(pos_mask*sim, 1)
    neg_sim = torch.sum(neg_mask*sim, 1)

    loss = -(torch.mean(torch.log(pos_sim / (pos_sim + neg_sim))))
    return loss.to(reps.device)