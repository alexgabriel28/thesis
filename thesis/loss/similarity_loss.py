import torch

def similarity_loss(
                reps: torch.Tensor,
                labels: torch.Tensor,
                t: float,
                alpha: float,
                ) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert (alpha >= 0) & (alpha <=1), "Alpha must be in [0,1]"
    
    eps = torch.Tensor([1e-08]).to(device)
    Eps = torch.Tensor([1.797693134862315e+308]).to(device)
    v_1, v_2 = torch.sum(torch.square(reps), dim = 1).view(reps.size(0), 1), \
        torch.sum(torch.square(reps), dim = 1).view(1, reps.size(0))
    norm_matrix = torch.matmul(torch.sqrt(v_1), torch.sqrt(v_2))
    sim = torch.matmul(reps, reps.T)/(torch.max(eps, norm_matrix)*t)
    sim = (torch.clamp(torch.exp(sim + 0), min = eps, max = Eps)- torch.diag(torch.diagonal(sim))).view(-1)

    pos_mask = (~torch.abs(labels.T - labels).bool()).float().view(-1)
    neg_mask = (torch.abs(labels.T - labels).bool()).float().view(-1)

    pos_sim = pos_mask@sim
    neg_sim = neg_mask@sim
    pos_loss = -torch.sum(torch.log(alpha*pos_sim))
    neg_loss = -torch.sum(torch.log((1 - alpha)*neg_sim))
    loss =  (pos_loss - neg_loss)/2*labels.size(dim = 1)
    
    return loss.to(reps.device)