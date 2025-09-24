import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_sum
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from lightning import LightningModule
from satbench.models.gnn import GNN
from satbench.utils.utils import safe_log, safe_div


class GNNPL(LightningModule):
    def __init__(self, args):
        super().__init__()
        assert args.task == 'assignment', "Task must be assignment (returning SAT formula)."
        self.model = GNN(args)
        self.save_hyperparameters() # save args to self.hparams

    def configure_optimizers(self):
        args = self.hparams['args']
        optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler is not None:
            if args.scheduler == 'ReduceLROnPlateau':
                assert args.valid_dir is not None
                scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience)
            else:
                assert args.scheduler == 'StepLR'
                scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_factor)

            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def forward(self, data):
        return self.model(data)
    
    def count_sat(self, data, v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size):
        v_assign = (v_pred > 0.5).float()
        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
        sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()

        # To calculate accuracy during evaluation, select only the SAT problems
        sat_batch = sat_batch[data.sat_problem]
        
        return sat_batch

    def training_step(self, data, batch_idx):
        args = self.hparams['args']
        batch_size = data.num_graphs
        c_size = data.c_size.sum().item()
        c_batch = data.c_batch
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        v_pred = self.forward(data)

        if args.loss == 'supervised':
            label = data.y
            loss = F.binary_cross_entropy(v_pred, label)

        elif args.loss == 'unsupervised_1':
            # calculate the loss in Eq. 4 and Eq. 5
            v_pred = self.forward_unsup(data)
            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
            s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
            s_max_nom = l_pred[l_edge_index] * s_max_denom

            c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
            c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
            c_pred = safe_div(c_nom, c_denom)

            s_min_denom = (-c_pred / 0.1).exp()
            s_min_nom = c_pred * s_min_denom
            s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
            s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

            score = safe_div(s_nom, s_denom)
            loss = (1 - score).mean()

        elif args.loss == 'unsupervised_2':
            # calculate the loss in Eq. 6
            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
            c_loss = -safe_log(1 - l_pred_aggr.exp())
            loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()

        sat_batch = self.count_sat(data, v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size)
        self.log('loss_train', loss.detach().cpu().item(), on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log('acc_train', sat_batch.mean().detach().cpu().item(), on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, data, batch_idx):
        args = self.hparams['args']
        batch_size = data.num_graphs
        c_size = data.c_size.sum().item()
        c_batch = data.c_batch
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        v_pred = self.forward(data)

        if args.loss == 'supervised':
            label = data.y
            loss = F.binary_cross_entropy(v_pred, label)
        
        elif args.loss == 'unsupervised_1':
            # calculate the loss in Eq. 4 and Eq. 5
            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
            s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
            s_max_nom = l_pred[l_edge_index] * s_max_denom

            c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
            c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
            c_pred = safe_div(c_nom, c_denom)

            s_min_denom = (-c_pred / 0.1).exp()
            s_min_nom = c_pred * s_min_denom
            s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
            s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

            score = safe_div(s_nom, s_denom)
            loss = (1 - score).mean()

        elif args.loss == 'unsupervised_2':
            # calculate the loss in Eq. 6
            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
            c_loss = -safe_log(1 - l_pred_aggr.exp())
            loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()
        
        elif args.loss == 'unsupervised_3':
            # The PUBO equivalent of the above which I wrote down?
            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
            c_loss = safe_log(l_pred_aggr.exp())
            loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()

        sat_batch = self.count_sat(data, v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size)
        self.log('loss_test', loss.detach().cpu().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('acc_test', sat_batch.mean().detach().cpu().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)        
        
        torch.cuda.empty_cache()
        return loss.detach().cpu().item()