import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_sum
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from lightning import LightningModule
from satbench.models.gnn import GNN
from satbench.utils.utils import safe_log, safe_div
from adabelief_pytorch import AdaBelief


class GNNPL(LightningModule):
    def __init__(self, args):
        super().__init__()
        assert args.task == 'assignment', "Task must be assignment (returning SAT formula)."
        self.model = GNN(args)
        self.save_hyperparameters() # save args to self.hparams

    def configure_optimizers(self):
        args = self.hparams['args']
        optimizer = AdaBelief(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.6, 0.999)) 
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
        v_assign = (v_pred >= 0.5).float()
        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
        sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()

        # To calculate accuracy during evaluation, select only the SAT problems (Not used here - unsup only training)
        if not self.training:
            sat_batch = sat_batch[data.sat_problem]
        
        return sat_batch
    
    def energy_fn(self, data, v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size):
        v_assign = (v_pred >= 0.5).float()
        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
        c_unsat = 1 - c_sat 
        energy = scatter_sum(c_unsat, c_batch, dim=0, dim_size=batch_size)
        
        return energy

    def training_step(self, data, batch_idx):
        args = self.hparams['args']
        batch_size = data.num_graphs
        c_size = data.c_size.sum().item()
        c_batch = data.c_batch
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        assert args.loss == 'unsupervised_2', "QuerySAT only supports their loss equation"
        time_weights = torch.arange(1, self.model.opts.n_iterations + 1, dtype=torch.float32, device=c_batch.device)
        time_weights = torch.exp(time_weights * 0.2) # exponential weight scaling
        time_weights = time_weights / time_weights.sum()

        v_pred_t = self.forward(data)
        
        final_loss = 0.
        for i, v_pred in enumerate(v_pred_t):
            losses = []
            for j in range(v_pred.shape[1]):
                mode_pred = v_pred[:, j]
                l_pred = torch.stack([mode_pred, 1 - mode_pred], dim=1).reshape(-1)
                l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                c_loss = -safe_log(1 - l_pred_aggr.exp())
                loss = scatter_sum(c_loss, c_batch, dim=0)
                assert batch_size == loss.shape[-1], "the loss must be calculated for each element of the batch"
                losses.append(loss)

            losses = torch.stack(losses, dim=0)

            # Multi-objective loss
            u = losses.size(0)
            sorted_losses, indices = torch.sort(losses, dim=0, descending=True)
            weights = torch.arange(1, u + 1, device=losses.device, dtype=losses.dtype) ** 2
            weights = weights.unsqueeze(1).repeat_interleave(batch_size, dim=1)
            weights = torch.gather(weights, dim=0, index=indices)
            losses_train = torch.sum(sorted_losses * weights, dim=0) / torch.sum(weights, 0)
            final_loss += losses_train.mean() * time_weights[j] # Minimize loss over timesteps

            # Get the best assignments by manual slicing
            problems = v_pred.split((data.l_size // 2).tolist(), dim=0)
            best_pred_idx = torch.argmin(losses, dim=0)
            assert len(problems) == best_pred_idx.size(0) == batch_size, "these should correspond if batching is correct!"
            
            if i == len(v_pred_t)-1: # Data for training energy can be calculated only at the last timestep
                final_v_pred = []
                for i in range(batch_size):
                    best = problems[i][:, best_pred_idx[i]]
                    final_v_pred.append(best)
                final_v_pred = torch.cat(final_v_pred, dim=0)

        energies = self.energy_fn(data, final_v_pred, l_edge_index, c_edge_index, c_size, c_batch, batch_size)
        self.log('loss_train', final_loss.detach().cpu().item(), on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log('energy_train', energies.mean().detach().cpu().item(), on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        
        return final_loss
    
    @torch.inference_mode(False) # Enable gradient computation for validation - the query mechanism requires it
    def validation_step(self, data, batch_idx):
        torch.set_grad_enabled(True)

        args = self.hparams['args']
        batch_size = data.num_graphs
        c_size = data.c_size.sum().item()
        c_batch = data.c_batch
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        assert args.loss == 'unsupervised_2', "QuerySAT only supports their loss equation"
        
        # During evaluation we only take the result of the last iteration
        v_pred = self.forward(data)
        losses = []
        for i in range(v_pred.shape[1]):
            mode_pred = v_pred[:, i]
            l_pred = torch.stack([mode_pred, 1 - mode_pred], dim=1).reshape(-1)
            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
            c_loss = -safe_log(1 - l_pred_aggr.exp())
            loss = scatter_sum(c_loss, c_batch, dim=0)
            assert batch_size == loss.shape[-1], "the loss must be calculated for each element of the batch"
            losses.append(loss)

        losses = torch.stack(losses, dim=0)
        best_pred_idx = torch.argmin(losses, dim=0)
        best_losses = losses.gather(dim=0, index=best_pred_idx.long().view(1,batch_size))
        loss_test = best_losses.mean()

        # Get the best assignments by manual slicing
        problems = v_pred.split((data.l_size // 2).tolist(), dim=0)
        assert len(problems) == best_pred_idx.size(0) == batch_size, "these should correspond if batching is correct!"
        
        final_v_pred = []
        for i in range(batch_size):
            best = problems[i][:, best_pred_idx[i]]
            final_v_pred.append(best)
        final_v_pred = torch.cat(final_v_pred, dim=0)

        self.log('loss_test', loss_test.detach().cpu().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        torch.cuda.empty_cache()
        return loss_test.detach().cpu().item()