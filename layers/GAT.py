class GATAttentionHead(torch.nn.Module):
  def __init__(self, sz_in, sz_out):
      super().__init__()
      self.sz_out = sz_out
      self.sz_in = sz_in

      # To linearly transform features
      # Initialization params from the original paper
      self.W = nn.Parameter(torch.empty(size=(sz_in, sz_out))) # F X F'
      nn.init.xavier_uniform_(self.W.data, gain=1.414)

      # To calculate attention coefficients
      # Initialization params from the original paper
      self.a = nn.Parameter(torch.empty(size=(2*sz_out, 1))) # 2F' X 1
      nn.init.xavier_uniform_(self.a.data, gain=1.414)
      self.leaky_relu = nn.LeakyReLU(0.2)
  
  def forward(self, fts, adj):
      # Apply the linear transformation
      new_fts = torch.mm(fts, self.W)
      alphas = torch.full_like(adj, -9e15)

      # Concatenate new feature vectors for each present edge
      edges = adj.nonzero(as_tuple=True) # ([u1, u2, ...], [v1, v2, ....])
      for i in range(edges.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])

        F.softmax(e_full, dim=1)
      a_input = torch.cat([new_fts[edges[0]], new_fts[edges[1]]], dim=1) # |E| x 2F'

      # Apply the 1-layer MLP a to get the unnormalized scores
      e = self.leaky_relu(torch.mm(a_input, self.a).squeeze())

      # Place the scores in an NxN matrix
      # Set all other scores to a very small value for softmax
      e_full = torch.full_like(adj, -9e15)
      e_full[edges] = e

      # Compute the attention coefficients (will be 0 for non-edges)
      alphas = F.softmax(e_full, dim=1)

      # Finally use the alphas to weight the incoming features
      ret_fts = torch.matmul(alphas, new_fts)
      return ret_fts

class GATLayer(torch.nn.Module):
  def __init__(self, num_heads, sz_in, sz_out, agg):
      super().__init__()

      # Aggregation ('concat', or 'mean' for the last layer)
      self.agg = agg

      # Independent attention heads
      heads = []
      for _ in range(num_heads):
          heads.append(GATAttentionHead(sz_in, sz_out))
      self.heads = nn.ModuleList(heads)

  def forward(self, fts, adj):
      # Apply all heads
      outs = [head(fts, adj) for head in self.heads]
      if self.agg == 'concat':
          # Apply the activation and concatenate
          # N x num_heads*sz_out
          return torch.cat([F.elu(out) for out in outs], dim=1)
      else:
          # No activation, average (assumes we average only at the end)
          # N x sz_out
          return torch.mean(torch.stack(outs), dim=0)