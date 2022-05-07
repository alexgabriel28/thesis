import torch

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