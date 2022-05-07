def energy_loss(
    reps: torch.Tensor = None, 
    labels: torch.Tensor = None,
    prototypes: torch.Tensor = None,
    alpha: float = 0.3,
    metric: str = "euclid",
    warm_up: int = 20,
    epoch: int = None,
    counts: int = None,
    num_classes: int = 3,
    t: float = 0.3,
    m: float = 0.5,
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_size = reps.size()[1]    
    eps = torch.Tensor([1e-08]).to(device)
    Eps = torch.Tensor([1.797693134862315e+308]).to(device)

    def euclid(reps, labels, prototypes):
        # Calculate the euclidean distance by binomial expansion:
        # (a - b)**2 = a**2 + 2ab + b**2
        # vector --> a.T*a + 2*a.T*b + b.T * b = (a - b)^2
        loss = torch.Tensor([0]).to(device)
        if prototypes is None:
            cosine_matrix = reps @ reps.T
            reps_squared = reps.pow(2).sum(1)
            diff = reps_squared - reps_squared.T
            dist_matrix = (cosine_matrix + diff)/feature_size

            pos_mask = (~torch.abs(labels.T - labels).bool()).float()
            neg_mask = (torch.abs(labels.T - labels).bool()).float()
            pos_loss = alpha*(pos_mask*dist_matrix).mean()
            neg_loss = (1-alpha)*(neg_mask*dist_matrix).mean()
            loss = pos_loss - neg_loss
            return loss

        else:
            cosine_matrix = reps @ prototypes.T
            reps_squared = reps.pow(2).sum(1, keepdim = True)
            prototypes_squared = prototypes.pow(2).sum(1, keepdim = True)
            prototypes_diff = prototypes_squared - prototypes_squared.T

            cosine_matrix_neg = reps @ reps.T
            diff_neg = reps_squared - reps_squared.T
            dist_matrix_neg = (cosine_matrix_neg + diff_neg)/feature_size

            prototypes_squared = prototypes_squared.repeat(1, reps.size()[0])
            reps_squared = reps_squared.repeat(1, prototypes.size()[0])

            diff = prototypes_squared.T - reps_squared
            dist_matrix = (cosine_matrix + diff)/feature_size

            pos_mask_ohe = torch.nn.functional.one_hot(labels).to(device)
            #neg_mask_ohe = (~pos_mask_ohe.bool()).float().squeeze()
            neg_mask = (torch.abs(labels.T - labels).bool()).float()

            t = reps.size()[0]
            pos_loss = alpha*(pos_mask_ohe*dist_matrix).mean()
            neg_loss = (1-alpha)*(neg_mask*dist_matrix_neg).mean()
            loss = pos_loss - neg_loss

            prototypes_cosine = prototypes @ prototypes.T
            prototypes_dist = (prototypes_cosine + prototypes_diff).mean()
            return loss - 0.01*prototypes_dist

    def cosine(reps, labels, prototypes, t, m):
        # Calculate the cosine similarity of reps / prototypes
        loss = torch.Tensor([0]).to(device)

        if prototypes is None:
            cosine_matrix = reps @ reps.T
            reps_squared = reps.pow(2).sum(1)
            norm_matrix = (reps_squared.pow(0.5) @ reps_squared.pow(0.5).T).to(device)

            sim_matrix = cosine_matrix/norm_matrix
            sim_matrix = torch.exp((sim_matrix - torch.diag(torch.diagonal(sim_matrix)))*t)
            #sim_matrix = torch.clamp(torch.exp(sim_matrix/t + 0), min = eps, max = Eps)

            pos_mask = (~torch.abs(labels.T - labels).bool()).float()
            neg_mask = (torch.abs(labels.T - labels).bool()).float()
            pos_loss = alpha*(1-pos_mask*sim_matrix).mean()
            neg_loss = (1-alpha)*torch.max(torch.Tensor([0]).to(device), neg_mask*sim_matrix - m).mean()

            loss = pos_loss + neg_loss

            return loss

        else:
            cosine_matrix = reps @ prototypes.T
            reps_squared = reps.pow(2).sum(1, keepdim = True)
            prototypes_squared = prototypes.pow(2).sum(1, keepdim = True)
            norm_matrix = (reps_squared.pow(0.5) @ prototypes_squared.pow(0.5).T).to(device)
            sim_matrix = cosine_matrix/norm_matrix
            #sim_matrix = torch.clamp(torch.exp(sim_matrix/t + 0), min = eps, max = Eps)

            pos_mask = torch.nn.functional.one_hot(labels).to(device)
            neg_mask = (~pos_mask.bool()).float()

            pos_loss = alpha*(1-(torch.exp(pos_mask*sim_matrix))).mean()
            neg_loss = (1-alpha)*torch.max(torch.Tensor([0]).to(device), neg_mask*sim_matrix - m)
            neg_loss = (t*torch.exp(neg_loss*(1/t))).mean()

            prototypes_cosine = torch.mean(prototypes @ prototypes.T)
            
            loss = pos_loss + neg_loss + prototypes_cosine*0.01

            return loss

    def update_proto(reps, labels, prototypes, counts):
            label_copy = labels.detach().cpu().numpy()
            prototypes_updated = torch.Tensor().to(device)

            for i, label in enumerate(np.unique(label_copy)):
                mask = torch.Tensor(np.array(label == label_copy).astype(float)).to(device)
                label = int(label)
                prototype = mask.T @ reps
                counts[label] += mask.sum()
                prototypes_updated = torch.cat((prototypes_updated, (1-mask.sum()/counts[label])*prototypes[label]+(mask.sum()/counts[label])*prototype))
            return prototypes_updated, counts

    if epoch == 0:
        prototypes = torch.zeros(size = (num_classes, int(feature_size))).to(device)
        counts = torch.zeros(prototypes.size()[0]).to(device)

    if epoch < warm_up:
        if metric == "euclid":
            loss = euclid(reps, labels, None)
        elif metric == "cosine":
            loss = cosine(reps, labels, None, t, m)

    else:
        if epoch == warm_up:
            prototypes, counts = update_proto(reps, labels, prototypes, counts)
        if metric == "euclid":
            loss = euclid(reps, labels, prototypes)
        elif metric == "cosine":
            loss = cosine(reps, labels, prototypes, t, m)
        prototypes, counts = update_proto(reps, labels, prototypes, counts)

    return loss, prototypes, counts