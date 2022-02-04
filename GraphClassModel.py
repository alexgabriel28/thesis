import torch.nn
import torch_geometric

class GraphClassificationModel(torch.nn.Module):

    def __init__(self, layer_type, num_layers=5, sz_in=3, sz_hid=256, sz_out=256):
        super().__init__()

        # GNN layers with ReLU, as before
        layers = []
        layers.append(layer_type(sz_in, sz_hid))
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(layer_type(sz_hid, sz_hid))
            layers.append(nn.ReLU())
        layers.append(layer_type(sz_hid, sz_hid))
        self.layers = nn.ModuleList(layers)

        # Final classifier
        self.f = nn.Linear(sz_hid, sz_out)
    
    def forward(self, fts, adj, batch):
        # 1: obtain node latents
        for l in self.layers:
            if isinstance(l, nn.ReLU):
                fts = l(fts)
            else:
                fts = l(fts, adj)

        # 2: pool
        h = torch_geometric.nn.global_mean_pool(fts, batch)

        # 3: final classifier
        return self.f(h)