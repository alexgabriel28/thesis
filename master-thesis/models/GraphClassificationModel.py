import torch.nn as nn
import torch_geometric

class GraphClassificationModel(torch.nn.Module):
    def __init__(self, 
                 layer_type, 
                 num_layers=5, 
                 num_heads = 1,
                 edge_dim = 3,
                 sz_in=3, 
                 sz_hid=64, 
                 sz_out=1024,
                 add_self_loops = True,
                 concat = False,
                 dropout = 0.3
                 ):
        super().__init__()

        self.sz_hid = sz_hid

        # GNN layers with ReLU, as before
        layers = []
        layers.append(layer_type(in_channels = sz_in, out_channels = sz_hid, heads = num_heads, edge_dim = edge_dim, dropout = dropout, concat= concat, add_self_loops = add_self_loops))
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(layer_type(in_channels = self.sz_hid, out_channels = self.sz_hid*2, heads = num_heads, edge_dim = edge_dim, dropout = dropout, concat= concat, add_self_loops = add_self_loops))
            layers.append(nn.ReLU())
            self.sz_hid *= 2
        layers.append(layer_type(in_channels = self.sz_hid, out_channels = 2*sz_hid, heads = num_heads, edge_dim = edge_dim, dropout = dropout, concat= concat, add_self_loops = add_self_loops))
        self.layers = nn.ModuleList(layers)
        self.sz_hid *= 2

        # Final classifier
        self.fc = nn.Linear(sz_hid, sz_out)
        self.f = nn.Linear(32, 3)
        
        #Use only for supervised classification
        #self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # 1: obtain node latents
        for l in self.layers:
            if isinstance(l, nn.ReLU):
                x = l(x)
            else:
                x = l(x, edge_index, edge_attr)

        # 2: pool
        h = torch_geometric.nn.global_add_pool(x, batch)
        h = self.fc(h)

        # 3: final classifier
        #Use only for supervised classification
        #return self.softmax(self.f(h))

        return h