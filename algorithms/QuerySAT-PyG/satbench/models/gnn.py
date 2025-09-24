import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from satbench.models.mlp import MLP
from torch_geometric.nn import PairNorm
from torch_scatter import scatter_sum, scatter_mean
from satbench.utils.utils import safe_log, safe_div



class QuerySAT(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        # Constants
        self.num_assignments = 8
        self.c_loss_scale = 4.


        self.variables_norm = PairNorm()
        self.clauses_norm = PairNorm()

        self.variables_query = MLP(2, self.opts.dim + 4, self.opts.dim, 1, self.opts.activation)
        self.clause_func = MLP(2, self.opts.dim + 1, self.opts.dim, self.opts.dim, self.opts.activation)
        self.variable_func = MLP(3, (self.opts.dim * 3) + 5, self.opts.dim, self.opts.dim, self.opts.activation)
        self.predictor = MLP(2, self.opts.dim, self.opts.dim, self.num_assignments, self.opts.activation)

    def complete_loss(self, v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size):
        # calculate the loss in Eq. 6
        l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
        l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
        c_loss = -safe_log(1 - l_pred_aggr.exp())
        loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).sum()

        return loss, c_loss
    
    def softplus_loss(self, v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size):
        l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
        l_pred_aggr = scatter_sum(F.softplus(l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
        c_loss = torch.exp(-l_pred_aggr)
        loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).sum()

        return loss, c_loss

    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb, batch_size, c_batch, l_batch):
        v_emb, _ = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
        v_batch, _ = torch.chunk(l_batch.reshape(l_size // 2, -1), 2, 1)
        
        # Scaling factors based on node degrees
        lit_deg = scatter_sum(torch.ones_like(l_edge_index), l_edge_index, dim=0, dim_size=l_size)
        lit_deg_weight = torch.rsqrt(torch.clamp(lit_deg.float(), min=1.0))
        var_deg = lit_deg.view(2, l_size//2).sum(dim=0)
        var_degree_weight = 4.0 * torch.rsqrt(torch.clamp(var_deg.float(), min=1.0))
        
        if self.training:
            v_embs, c_embs, assignments = [], [], []

        for _ in range(self.opts.n_iterations):
            # produce queries     
            v_in = torch.cat([v_emb, torch.randn([l_size//2, 4]).to(l_emb.device)], axis=-1)
            query = torch.sigmoid(self.variables_query(v_in))
            loss, c_loss = self.softplus_loss(query.view(-1), l_edge_index, c_edge_index, c_size, c_batch, batch_size)

            # compute gradient w.r.t. query
            query_grad = torch.autograd.grad(
                loss, 
                query, 
                create_graph=True
            )[0]
            query_grad = query_grad * var_degree_weight.unsqueeze(1)  # scale

            # scale the clause loss
            c_loss = c_loss * self.c_loss_scale

            # update clause embeddings
            c_in = torch.cat([c_emb, c_loss.view(-1, 1)], dim=-1)
            c_emb = self.clauses_norm(self.clause_func(c_in), c_batch, batch_size)

            # update variable embeddings
            c2l_msg = c_emb[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            c2l_msg_aggr = c2l_msg_aggr * lit_deg_weight.unsqueeze(1)
            c2v_msg = c2l_msg_aggr.reshape(l_size//2, -1)
            v_temp = torch.cat([v_in, c2v_msg, query_grad], dim=-1)
            v_upd = self.variables_norm(self.variable_func(v_temp), v_batch.squeeze(), batch_size)
            v_emb = v_upd + (0.1 * v_in[:, :-4])

            if self.training:
                c_embs.append(c_emb)
                v_embs.append(v_emb)

            # produce assignments
            out = torch.sigmoid(self.predictor(v_emb))

            if self.training:
                assignments.append(out)
            
            # del query_grad # free memory

        if self.training:
            return v_embs, c_embs, assignments
        else:
            return v_emb, c_emb, out

class GNN(nn.Module):
    def __init__(self, opts):
        super(GNN, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.l_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
        
        self.gnn = QuerySAT(self.opts)
    
    def forward(self, data):
        batch_size = data.num_graphs
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index
        device = c_edge_index.device
        c_batch = data.c_batch
        l_batch = data.l_batch

        assert self.opts.init_emb == 'ones', "QuerySAT starts with an all ones vector representation"
        assert self.opts.task == 'assignment', 'Only assignment task is currently supported in QuerySAT'

        # Initialize literal and clause embeddings
        if self.opts.init_emb == 'learned':
            l_emb = (self.l_init).repeat(l_size, 1)
            c_emb = (self.c_init).repeat(c_size, 1)
        elif self.opts.init_emb == 'ones':
            l_emb = torch.ones(l_size, self.opts.dim, device=device)
            c_emb = torch.ones(c_size, self.opts.dim, device=device)
        else:
            l_emb = torch.randn(l_size, self.opts.dim, device=device) * math.sqrt(2 / self.opts.dim)
            c_emb = torch.randn(c_size, self.opts.dim, device=device) * math.sqrt(2 / self.opts.dim)

        _, _, assignments = self.gnn(l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb, batch_size, c_batch, l_batch)

        return assignments

