import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from modules import AllConv2dModule, SepConv1dModule
from modules import FreqUnrollBlock, ACTBlock, DACTModule, DACTBlock
#  from metrics import F1_Ours
from thop import profile
import pdb


class LitModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # global feature extractor
        if not params['model']['enc_type']:
            self.encoder = None
        elif params['model']['enc_type'] == 'sep1d':
            self.encoder = SepConv1dModule(**params['model']['enc_params'])
        elif params['model']['enc_type'] == 'all2d':
            self.encoder = AllConv2dModule(**params['model']['enc_params'])
        else:
            raise NotImplementedError("encoder type {} not supported".format(params['model']['enc_type']))
        # model type
        self.block_type = params['model']['block_type']
        if self.block_type == 'ACT':
            params['model']['cell_params']['multiclass'] = params['model']['multiclass']
            self.nn = ACTBlock(params['model']['cell_params'])
        elif self.block_type == 'l2_naive':
            params['model']['cell_params']['multiclass'] = params['model']['multiclass']
            self.nn = FreqUnrollBlock(**params['model']['cell_params'])
            assert 'conf_params' in params['model'], "confidence params needs to be configured"
            self.env_type = params['model']['conf_params']['env_type']
            if self.env_type not in ['linear']:
                raise NotImplementedError("Envelope type {} not implemented".format(self.env_type))
            self.env_args = params['model']['conf_params']['env_args']
        elif self.block_type == 'FU_baseline':
            params['model']['cell_params']['multiclass'] = params['model']['multiclass']
            self.nn = FreqUnrollBlock(**params['model']['cell_params'])
        elif self.block_type == 'DACT':
            params['model']['cell_params']['multiclass'] = params['model']['multiclass']
            self.nn = DACTModule(**params['model']['cell_params'])
        elif self.block_type == 'DACT_ours':
            params['model']['cell_params']['multiclass'] = params['model']['multiclass']
            self.nn = DACTBlock(params['model']['cell_params'])
        else:
            raise NotImplementedError("block type {} is not supported".format(self.block_type))
        self.multiclass = params['model']['multiclass']
        self.n_class = params['model']['cell_params']['n_class']
        # loss functions
        if self.multiclass:
            if params['model']['loss_type'] == 'NLLLoss':
                self.loss_fn = nn.NLLLoss()
            elif params['model']['loss_type'] == 'CrossEntropy':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError("loss type is not supported")
        else:
            assert params['model']['loss_type'] == 'BCELoss', "multilabel classification requires BCELoss"
            self.loss_fn = nn.BCELoss()
        self.use_thop = params['model']['use_thop']
        self.tau = params['model']['tau']
        # metrics
        self.train_metrics = nn.ModuleDict({
            'acc': pl.metrics.Accuracy(),
            'f1': pl.metrics.F1(num_classes=self.n_class, average='macro'),
        })
        self.valid_metrics = nn.ModuleDict({
            'acc': pl.metrics.Accuracy(),
            'f1': pl.metrics.F1(num_classes=self.n_class, average='macro'),
        })
        self.test_metrics = nn.ModuleDict({
            'acc': pl.metrics.Accuracy(),
            'f1': pl.metrics.F1(num_classes=self.n_class, average='macro'),
        })

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        out = self.nn(x)
        return out

    def training_step(self, batch, idx):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, _ = batch
        if self.multiclass:
            y = y.argmax(-1)  # for multiclass, use int labels
        else:
            y = y.float()
        if self.block_type == 'ACT':
            y_pred, N_cost, R_cost = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            self.log('train_task_loss', task_loss)
            ponder_cost = N_cost + R_cost
            ponder_loss = ponder_cost.mean()
            self.log('train_num_cost', N_cost.mean())
            self.log('train_ponder_loss', ponder_loss)
            loss = task_loss + self.tau * ponder_loss
        elif self.block_type == 'l2_naive':
            y_pred, conf = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
            task_loss = self.loss_fn(y_pred, y)
            self.log('train_task_loss', task_loss)
            if self.env_type == 'linear':
                self.env_args['steps'] = conf.shape[-1]
                env = torch.linspace(**self.env_args).to(conf.device)
            conf_loss = (env * (1 - conf)).mean()
            self.log('train_conf_loss', conf_loss)
            loss = task_loss + self.tau * conf_loss
        elif self.block_type == 'FU_baseline':
            y_pred = self.forward(x)
            y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
            task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
        elif self.block_type == 'DACT':
            y_pred, ponder_cost = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            self.log('train_task_loss', task_loss)
            ponder_loss = ponder_cost.mean()
            self.log('train_ponder_loss', ponder_loss)
            loss = task_loss + self.tau * ponder_loss
        elif self.block_type == 'DACT_ours':
            y_pred, ponder_cost = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            self.log('train_task_loss', task_loss)
            ponder_loss = ponder_cost.mean()
            self.log('train_ponder_loss', ponder_loss)
            loss = task_loss + self.tau * ponder_loss
        else:
            raise NotImplementedError("block type not supported")
        self.log('train_loss', loss)
        for met in self.train_metrics.keys():
            met_fn = self.train_metrics[met]
            if self.multiclass:
                self.log('train_{}'.format(met), met_fn(y_pred.argmax(1), y))
            else:
                self.log('train_{}'.format(met), met_fn(y_pred, y.long()))
        return loss

    def training_epoch_end(self, outputs):
        for met in self.train_metrics.keys():
            met_fn = self.train_metrics[met]
            met_fn.compute()
            met_fn.reset()

    def validation_step(self, batch, idx):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, _ = batch
        if self.multiclass:
            y = y.argmax(-1)  # for multiclass, use int labels
        else:
            y = y.float()
        if self.block_type == 'ACT':
            y_pred, N_cost, R_cost = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            self.log('valid_task_loss', task_loss)
            ponder_cost = N_cost + R_cost
            ponder_loss = ponder_cost.mean()
            self.log('valid_num_step', N_cost.mean())
            self.log('valid_ponder_loss', ponder_loss)
            loss = task_loss + self.tau * ponder_loss
        elif self.block_type == 'l2_naive':
            y_pred, conf, n_steps = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
            task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
            self.log('valid_num_step', n_steps.mean())
        elif self.block_type == 'FU_baseline':
            y_pred = self.forward(x)
            y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
            task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
        elif self.block_type == 'DACT':
            y_pred, n_steps = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
            self.log('valid_num_step', n_steps.mean())
        elif self.block_type == 'DACT_ours':
            y_pred, n_steps = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
            self.log('valid_num_step', n_steps.mean())
        else:
            raise NotImplementedError("block type not supported")
        self.log('valid_loss', loss, prog_bar=True)
        for met in self.valid_metrics.keys():
            met_fn = self.valid_metrics[met]
            if self.multiclass:
                met_fn.update(y_pred.argmax(1), y)
            else:
                met_fn.update(y_pred, y.long())

    def validation_epoch_end(self, outputs):
        for met in self.valid_metrics.keys():
            met_fn = self.valid_metrics[met]
            self.log('valid_{}'.format(met), met_fn.compute())
            met_fn.reset()

    def test_step(self, batch, idx):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, _ = batch
        if self.multiclass:
            y = y.argmax(-1)  # for multiclass, use int labels
        else:
            y = y.float()
        if self.block_type == 'ACT':
            y_pred, N_cost, R_cost = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            self.log('test_task_loss', task_loss)
            ponder_cost = N_cost + R_cost
            ponder_loss = ponder_cost.mean()
            self.log('test_num_step', N_cost.mean())
            self.log('test_ponder_loss', ponder_loss)
            loss = task_loss + self.tau * ponder_loss
        elif self.block_type == 'l2_naive':
            y_pred, conf, n_steps = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
            task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
            self.log('test_num_step', n_steps.mean())
        elif self.block_type == 'FU_baseline':
            y_pred = self.forward(x)
            y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
            task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
        elif self.block_type == 'DACT':
            y_pred, n_steps = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
            self.log('test_num_step', n_steps.mean())
        elif self.block_type == 'DACT_ours':
            y_pred, n_steps = self.forward(x)
            if self.multiclass:
                y_pred = y_pred.permute(0, 2, 1)  # for multiclass, (N, C, ...)
                task_loss = self.loss_fn(torch.log(y_pred), y)
            else:
                task_loss = self.loss_fn(y_pred, y)
            loss = task_loss
            self.log('test_num_step', n_steps.mean())
        else:
            raise NotImplementedError("block type not supported")
        self.log('test_loss', loss, prog_bar=True)
        if self.use_thop:
            macs, _ = profile(self, inputs=(x,))
            self.log('test_macs', macs)
        for met in self.test_metrics.keys():
            met_fn = self.test_metrics[met]
            if self.multiclass:
                met_fn.update(y_pred.argmax(1), y)
            else:
                met_fn.update(y_pred, y.long())

    def test_epoch_end(self, outputs):
        for met in self.test_metrics.keys():
            met_fn = self.test_metrics[met]
            self.log('test_{}'.format(met), met_fn.compute())
            met_fn.reset()

    def configure_optimizers(self):
        if self.params is None:
            opt = optim.Adam(self.parameters(), lr=3e-4)
            scheduler = optim.lr_scheduler.MultiplicativeLR(opt, 0.99)
        else:
            opt = getattr(optim, self.params['optimizer']['fn'])(self.parameters(), lr=self.params['optimizer']['lr'])
            scheduler = {
                'scheduler': optim.lr_scheduler.MultiStepLR(opt, **(self.params['scheduler'])),
                'interval': 'epoch',
            }

        return [opt], [scheduler]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
