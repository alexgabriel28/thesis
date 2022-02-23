###################################################
######### Graph Classification Model PNA ##########
###################################################

import torch.nn as nn
import torch_geometric
from torch_geometric.utils import degree
class GraphClassificationModel(nn.Module):
    def __init__(self, 
                 layer_type = torch_geometric.nn.PNAConv, 
                 num_layers=6,
                 mult_factor = 2, 
                 sz_in=3, 
                 sz_hid=64, 
                 sz_out=2048,
                 add_self_loops = True,
                 dropout = 0.3,
                 pre_layers = 1,
                 towers =1,
                 divide_input = False,
                 classification = False
                 ):
        super().__init__()
        torch.manual_seed(3407)
        self.sz_hid = sz_hid

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        layers = []
        layers.append(
            layer_type(
                in_channels = sz_in, 
                out_channels = sz_hid,
                aggregators = aggregators,
                scalers = scalers,
                deg = deg,
                pre_layers = pre_layers,
                divide_input = divide_input,
                towers = towers,
                )
            )
        layers.append(nn.ReLU())

        for _ in range(num_layers-2):
            layers.append(
                layer_type(
                    in_channels = sz_hid, 
                    out_channels = sz_hid,
                    aggregators = aggregators,
                    scalers = scalers,
                    deg = deg,
                    pre_layers = pre_layers,
                    divide_input = divide_input,
                    towers = towers,
                    )
                )
            
            layers.append(nn.ReLU())

        layers.append(
            layer_type(
                in_channels = sz_hid, 
                out_channels = sz_hid,
                aggregators = aggregators,
                scalers = scalers,
                deg = deg,
                pre_layers = pre_layers,
                divide_input = divide_input,
                towers = towers,
                )
            )
        
        self.layers = nn.ModuleList(layers)

        # Final classifier
        self.fc = nn.Linear(sz_hid, sz_hid)
        self.f = nn.Linear(sz_hid, sz_out)
        self.classification = classification
        self.dropout = nn.Dropout(p = dropout)

        #Use only for supervised classification
        if self.classification == True:
          self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, x, edge_index, batch):
        # 1: obtain node latents
        for l in self.layers:
            if isinstance(l, nn.ReLU):
                x = l(x)
            else:
                x = l(x, edge_index)

        # 2: pool
        h = torch_geometric.nn.global_add_pool(x, batch)
        h = self.fc(h.to(device))
        h = self.dropout(h)
        h = self.f(h)


        # 3: final classifier
        #Use only for supervised classification
        if self.classification == True:
          return self.softmax(h)
        else:
          return h

#Calculate degree for PNA
def calculate_degree(train_dataset):
  max_degree = -1
  for data, graph in train_dataset:
      d = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
      max_degree = max(max_degree, int(d.max()))
  # Compute the in-degree histogram tensor
  deg = torch.zeros(max_degree + 1, dtype=torch.long)
  for data, graph in train_dataset:
      d = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
      deg += torch.bincount(d, minlength=deg.numel())
  return max_degree, deg