from torch_geometric.utils import to_dense_adj

## TODO: implement me! :)
class GraphConvolution(torch.nn.Module):
  def __init__(self, num_input_features, num_output_features):
    super().__init__()
    self.W1 = torch.nn.Parameter(torch.empty(size = (num_input_features, num_output_features)))
    torch.nn.init.xavier_uniform_(self.W1)
    self.W2 = torch.nn.Parameter(torch.empty(size = (num_input_features, num_output_features)))
    torch.nn.init.xavier_uniform_(self.W2)
    
  def forward(self, x, edge_index):
    adj = to_dense_adj(edge_index)
    neighbors_aggregation = adj @ x
    out = x @ self.W1 + neighbors_aggregation @ self.W2

    return out