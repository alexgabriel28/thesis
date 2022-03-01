import torch

def similarity_loss(
                reps: torch.Tensor,
                labels: torch.Tensor,
                t: float,
                alpha: float,
                ) -> torch.Tensor:

    assert (alpha >= 0) & (alpha <=1), "Alpha must be in [0,1]"
    
    eps = torch.Tensor([1e-08]).to(device)
    v_1, v_2 = torch.sum(torch.square(reps), dim = 1).view(reps.size(0), 1), torch.sum(torch.square(reps), dim = 1).view(1, reps.size(0))
    norm_matrix = torch.matmul(torch.sqrt(v_1), torch.sqrt(v_2))
    #print(norm_matrix[norm_matrix == 0])
    sim = torch.matmul(reps, reps.T)/(torch.max(eps, norm_matrix)*t)
    #print(torch.isinf(sim).any())
    sim = (torch.exp(sim + 0) - torch.diag(torch.diagonal(sim))).view(-1)
    #print(torch.isnan(sim).any())

    pos_mask = (~torch.abs(labels.T - labels).bool()).float().view(-1)
    neg_mask = (torch.abs(labels.T - labels).bool()).float().view(-1)

    pos_sim = pos_mask@sim
    neg_sim = neg_mask@sim
    #print(torch.isnan(sim).any())
    pos_loss = alpha*pos_sim.sum()
    neg_loss = (1 - alpha)*neg_sim.sum()
    loss =  neg_loss - pos_loss
    
    return loss.to(reps.device)