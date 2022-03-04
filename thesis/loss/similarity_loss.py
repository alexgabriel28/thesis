import torch

def cosine_sim(
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
    sim = sim - torch.diag(torch.diagonal(sim))
    sim = torch.clamp(torch.exp(sim + 0), min = eps, max = Eps)

    pos_mask = (~torch.abs(labels.T - labels).bool()).float()
    neg_mask = (torch.abs(labels.T - labels).bool()).float()

    pos_loss = torch.mean(alpha*(pos_mask*sim))
    neg_loss = torch.mean((1 - alpha)*(neg_mask*sim))
    loss =  (neg_loss - pos_loss)
    
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