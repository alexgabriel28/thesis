###############################################
######### Graph Classification Model ##########
###############################################
import torch
import torch.nn as nn
import torch_geometric

class GraphClassificationModel(nn.Module):
    def __init__(self, 
                 layer_type = torch_geometric.nn.GATv2Conv, 
                 num_layers=6,
                 mult_factor = 1, 
                 num_heads = 1,
                 edge_dim = None,
                 sz_in=3, 
                 sz_hid=64, 
                 sz_out=2048,
                 add_self_loops = True,
                 concat = True,
                 dropout = 0.3,
                 classification = False
                 ):
        super().__init__()
        torch.manual_seed(3407)
        self.sz_hid = sz_hid
        self.edge_dim = edge_dim

        # GNN layers with ReLU, as before
        if concat == True:
          self.num_heads_factor = num_heads
        else:
          self.num_heads_factor = 1
          
        layers = []
        layers.append(
            layer_type(
                in_channels = sz_in, 
                out_channels = sz_hid, 
                heads = num_heads, 
                edge_dim = edge_dim, 
                dropout = dropout, 
                concat= concat, 
                add_self_loops = add_self_loops
                )
            )
        layers.append(nn.ReLU())

        for _ in range(num_layers-2):
            layers.append(
                layer_type(
                    in_channels = self.sz_hid*self.num_heads_factor, 
                    out_channels = self.sz_hid*mult_factor, 
                    heads = num_heads, 
                    edge_dim = edge_dim, 
                    dropout = dropout, 
                    concat= concat, 
                    add_self_loops = add_self_loops
                    )
                )
            
            layers.append(nn.ReLU())
            self.sz_hid *= mult_factor

        layers.append(
            layer_type(
                in_channels = self.sz_hid*self.num_heads_factor,
                out_channels = self.sz_hid*mult_factor, 
                heads = num_heads, 
                edge_dim = edge_dim, 
                dropout = dropout, 
                concat= concat, 
                add_self_loops = add_self_loops
                )
            )
        
        self.layers = nn.ModuleList(layers)
        self.sz_hid *= mult_factor

        # Final classifier
        self.fc = nn.Linear(self.sz_hid*self.num_heads_factor, sz_out)
        self.f = nn.Linear(32, 3)
        self.classification = classification

        #Use only for supervised classification
        if self.classification == True:
          self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # 1: obtain node latents
        if self.edge_dim == None:
          for l in self.layers:
              if isinstance(l, nn.ReLU):
                  x = l(x)
              else:
                  x = l(x, edge_index)
        else:
          for l in self.layers:
              if isinstance(l, nn.ReLU):
                  x = l(x)
              else:
                  x = l(x, edge_index, edge_attr)

        # 2: pool
        h = torch_geometric.nn.global_add_pool(x, batch)
        h = self.fc(h.to(device))

        # 3: final classifier
        #Use only for supervised classification
        if self.classification == True:
          return self.softmax(h)
        else:
          return h