import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1, trainable=False, mode='add'):
        super(PositionalEncoding, self).__init__()
        import math
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [S, P]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe.requires_grad = trainable
        assert mode in ['cat', 'add']
        self.mode = mode

    def forward(self, x):
        '''
            x: tensor with shape [B, S, K, T]
        '''
        assert len(x.shape) == 4, "dimension of input not supported"
        bs = x.shape[0]
        n_step = x.shape[1]
        seq_len = x.shape[-1]
        pe = self.pe[:n_step].unsqueeze(0).unsqueeze(-1)
        if self.mode == 'add':
            assert x.shape[2] == self.d_model, "dimension of model mismatch"
            out = x + pe
        elif self.mode == 'cat':
            pe = pe.expand((bs, -1, -1, seq_len))
            out = torch.cat((x, pe), dim=2) # [B, S, K+P, T]
        return out

    def stepwise_forward(self, x, step):
        '''
            x: tensor with shape [B, K, T]
        '''
        assert len(x.shape) == 3, "dimension of the input not supported"
        bs = x.shape[0]
        seq_len = x.shape[-1]
        pe = self.pe[step].unsqueeze(0).unsqueeze(-1) # [1, P, 1]
        if self.mode == 'add':
            assert x.shape[1] == self.d_model, "dimension of model mismatch"
            out = x + pe
        elif self.mode == 'cat':
            pe = pe.expand((bs, -1, seq_len))
            out = torch.cat((x, pe), dim=1) # [B, K+P, T]
        return out


def test_pe():
    device = torch.device('cuda:0')
    seq_len = 100
    x = torch.randn(16, 8, 128, seq_len).to(device)

    d_model = 128
    max_len = 8
    pe_params = {
        'd_model': 16,
        'max_len': 8,
        'mode': 'cat',
    }
    PE = PositionalEncoding(**pe_params).to(device)
    out = PE(x)
    pdb.set_trace()


class ContextGating(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ContextGating, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''
            x: tensor with shape [B, C, F, T]
        '''
        h = self.layers(x)
        out = h * x
        return out


class AllConv2dModule(nn.Module):
    def __init__(self, channels, kernel_sizes, pooling_sizes, dropout):
        super().__init__()
        layers = []
        assert len(channels) == len(pooling_sizes), "number of layers mismatch"
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes for _ in channels]
        self.max_kernel_size = max(kernel_sizes)
        for i in range(len(channels)):
            in_channels = 1 if i == 0 else channels[i-1]
            out_channels = channels[i]
            kernel_size = kernel_sizes[i]
            padding = kernel_size // 2
            pooling_size = pooling_sizes[i]
            layers.append(
                nn.Sequential(
                    #  nn.InstanceNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding),
                    ContextGating(out_channels, kernel_size),
                    nn.AvgPool2d(pooling_size),
                    nn.Dropout(dropout)
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        '''
            x: input spectrogram with shape [B, F, T]
        '''
        if len(x.shape) == 4:
            assert x.shape[-1] == 1, "unsupported input channel {}".format(
                x.shape[-1])
            x = x.squeeze(-1)
        if len(x.shape) == 3:
            x = x.unsqueeze(1) # (B, 1, F, T)
        out = self.layers(x)
        out = out.view(out.shape[0], -1, out.shape[-1]) # (B, F, T)
        return out


def test_conv():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)

    cnn_params = {
        'channels': (16, 32, 64, 128, 128, 128, 128),
        'kernel_sizes': 3,
        'pooling_sizes': [(2, 1) for _ in range(7)],
        'dropout': 0.5,
    }
    net = AllConv2dModule(**cnn_params).to(device)
    out = net(x)
    pdb.set_trace()


def test_conv_debug():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)

    cnn_params = {
        'channels': (16, 32, 64, 128, 128, 128, 128),
        'kernel_sizes': 1,
        'pooling_sizes': [(2, 1) for _ in range(7)],
        'dropout': 0.5,
    }
    net = AllConv2dModule(**cnn_params).to(device)
    net.eval()
    with torch.no_grad():
        idx = 2
        out = net(x)
        tmp = out[0,:,idx]
        x_part = x[:, :, :idx+1]
        out_part = net(x_part)
        pdb.set_trace()
        tmp2 = out_part[0, :, 0]
    #  net = nn.Conv2d(1, 128, 1).to(device)
    #  net.eval()
    #  with torch.no_grad():
    #      out = net(x.unsqueeze(1))
    #      pdb.set_trace()
    #      x_part = x[:, :, :1]
    #      out_part = net(x_part.unsqueeze(1))


class SepConv1dModule(nn.Module):
    def __init__(self, in_channels, kernel_size, n_layers):
        super(SepConv1dModule, self).__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.InstanceNorm1d(in_channels))
            layers.append(
                nn.Conv1d(in_channels, in_channels, kernel_size,
                          padding=kernel_size//2, groups=in_channels)
            )
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        '''
            x: tensor with shape (B, F, T)
        '''
        if len(x.shape) == 4:
            assert x.shape[-1] == 1, "unsupported input channel {}".format(
                x.shape[-1])
            x = x.squeeze(-1)
        out = self.layers(x)
        return out


def test_conv1d():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)

    cnn_params = {
        'in_channels': 128,
        'kernel_size': 5,
        'n_layers': 2,
    }
    net = SepConv1dModule(**cnn_params).to(device)
    out = net(x)
    pdb.set_trace()


class FreqUnrollBlock(nn.Module):
    def __init__(self,
                 block_size,
                 input_size,
                 n_hidden,
                 n_layer,
                 n_class,
                 cnn_params=None,
                 pe_params=None,
                 subband_type='contiguous',
                 dropout=0,
                 threshold=0.5,
                 max_step=None,
                 rand_step=False,
                 mode='l2_naive',
                 out_type='last',
                 multiclass=True):
        super().__init__()
        self.block_size = block_size
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.gru = nn.GRU(input_size,
                          n_hidden,
                          n_layer,
                          batch_first=True,
                          dropout=dropout)
        self.n_class = n_class
        if multiclass:
            self.fc = nn.Linear(n_hidden, n_class)
        else:
            self.fc = nn.Sequential(
                nn.Linear(n_hidden, n_class),
                nn.Sigmoid(),
            )
        self.cnn = None
        if cnn_params is not None:
            self.cnn = AllConv2dModule(**cnn_params)
        self.pe = None
        if pe_params is not None:
            self.pe = PositionalEncoding(**pe_params)
        self.threshold = threshold
        self.max_step = max_step
        self.rand_step = rand_step
        assert mode in ['fixed_step', 'l2_naive']
        self.mode = mode
        self.multiclass = multiclass
        assert subband_type in ['contiguous', 'linear_gap']
        self.subband_type = subband_type
        assert out_type in ['last', 'conf_weighted']
        self.out_type = out_type

    def forward(self, x):
        '''
            x: input tensor with shape (B, F, T)
        '''
        if len(x.shape) == 4:
            assert x.shape[-1] == 1, "unsupported input channel {}".format(
                x.shape[-1])
            x = x.squeeze(-1)
        bs, n_f, n_t = x.shape
        n_sb = n_f // self.block_size  # number of subbands
        if self.subband_type == 'contiguous':
            x = x.view(bs, n_sb, self.block_size, n_t)  # (B, S, K, T)
        elif self.subband_type == 'linear_gap':
            x = x.view(bs, self.block_size, n_sb,
                       n_t).transpose(1, 2)  # (B, S, K, T)
        else:
            raise NotImplementedError("subband type {} not supported".format(
                self.subband_type))
        # chop by the step limit
        if isinstance(self.max_step, int):
            n_sb = min(self.max_step, n_sb)
            x = x[:, :n_sb]
        if self.mode == 'fixed_step':
            # unroll the frequency for each time step
            x = x.permute(0, 3, 1, 2).reshape(bs, -1,
                                              self.input_size)  # (B, TS, K')
            h, _ = self.gru(x)  # (B, TS, D)
            h = h.reshape(bs, n_t, n_sb, self.n_hidden)  # (B, T, S, D)
            if self.out_type == 'last':
                out = self.fc(h[:, :, -1])  # (B, T, C)
            else:
                raise NotImplementedError("output type not supported")
            return out
        elif self.mode == 'l2_naive':
            if self.training:
                # chop by the random step limit
                if self.rand_step:
                    n_sb = torch.randint(n_sb, (1,)).item() + 1
                    x = x[:, :n_sb]
                # processing with CNNs
                if self.cnn is not None:
                    x = x.reshape(bs*n_sb, -1, n_t) # (BS, K, T)
                    x = self.cnn(x) # (BS, K', T)
                    x = x.reshape(bs, n_sb, -1, n_t) # (B, S, K', T)
                if self.pe is not None:
                    x = self.pe(x) # (B, S, K+, T)
                # unroll the frequency for each time step
                x = x.permute(0, 3, 1,
                              2).reshape(bs, -1, self.input_size)  # (B, TS, K')
                h, _ = self.gru(x)  # (B, TS, D)
                h = h.reshape(bs, n_t, n_sb, self.n_hidden)  # (B, T, S, D)
                all_out = self.fc(h)  # (B, T, S, C)
                if self.multiclass:
                    prob = F.softmax(all_out, dim=-1)
                    conf = torch.linalg.norm(prob, dim=-1)**2  # (B, T, S)
                else:
                    # taking the difference between neutral and average across classes
                    conf = ((all_out - 0.5)**2).mean(dim=-1) # (B, T, S)
                if self.out_type == 'last':
                    out = all_out[:, :, -1]  # (B, T, C)
                elif self.out_type == 'conf_weighted':
                    conf_weight = F.softmax(conf, dim=-1) # (B, T, S)
                    out = all_out * conf_weight.unsqueeze(-1) # (B, T, S, C)
                    out = out.sum(dim=2) # (B, T, C)
                else:
                    raise NotImplementedError("output type not supported")
                return out, conf
            else:
                #  x = x.permute(0, 3, 1, 2)  # (B, T, S, K')
                out = torch.zeros(bs, n_t,
                                  self.n_class).to(x.device)  # (B, T, C)
                conf = torch.zeros(bs, n_t, n_sb).to(x.device)  # (B, T, S)
                n_steps = torch.zeros(bs, n_t).to(x.device)  # (B, T)
                hidden = torch.zeros(self.n_layer, bs,
                                     self.n_hidden).to(x.device)
                feat_dict = {}
                for t in range(n_t):
                    cont_mask = torch.ones(bs).to(x.device)
                    if self.out_type == 'conf_weighted':
                        out_t = torch.zeros(bs, n_sb, self.n_class).to(x.device) # (B, S, C)
                    for k in range(n_sb):
                        step_inp = x[:, k] # (B, K, T)
                        # getting features if applicable
                        if self.cnn is not None:
                            # proceed only if context window greater than 1
                            if self.cnn.max_kernel_size > 1:
                                if feat_dict.get(k, None) is None:
                                    step_inp = self.cnn(step_inp) # (B, K', T)
                                    if self.pe is not None:
                                        step_inp = self.pe.stepwise_forward(step_inp, k) # (B, K+, T)
                                    feat_dict[k] = step_inp
                                else:
                                    step_inp = feat_dict[k] # (B, K', T)
                        elif self.pe is not None:
                            if feat_dict.get(k, None) is None:
                                step_inp = self.pe.stepwise_forward(step_inp, k) # (B, K+, T)
                                feat_dict[k] = step_inp
                            else:
                                step_inp = feat_dict[k] # (B, K', T)
                        # process the current step
                        step_inp = step_inp[:, :, t] # (B, K')
                        # proceed only if context window is 1
                        if self.cnn is not None and self.cnn.max_kernel_size == 1:
                            step_inp = self.cnn(step_inp.unsqueeze(-1)) # (B, K', 1)
                            if self.pe is not None:
                                step_inp = self.pe.stepwise_forward(step_inp, k)
                            step_inp = step_inp.squeeze(-1) # (B, K')
                        # recurrent processing starts here
                        h_kt, hidden_kt = self.gru(step_inp.unsqueeze(1),
                                                   hidden)
                        # only update hidden states if not been processed
                        hidden = cont_mask[:, None] * hidden_kt + (1 - cont_mask)[:, None] * hidden  # [L, B, D]
                        z_kt = self.fc(h_kt).squeeze(1)  # (B, C)
                        if self.multiclass:
                            o_kt = F.softmax(z_kt, -1)
                            conf_kt = torch.linalg.norm(
                                o_kt, dim=-1)**2  # (B,) norm^2=1 <=> confidence
                        else:
                            conf_kt = ((z_kt - 0.5)**2).mean(dim=-1) # (B,)
                        # update conf and step history
                        conf[:, t, k] = conf_kt
                        n_steps[:, t] += cont_mask.detach()
                        """
                            update output only if
                                1) previous channel has not been processed
                            and 2) current prediction has high confidence
                                or 3) last channel
                        """
                        if k < n_sb - 1:
                            o_update = cont_mask * \
                                (conf_kt > self.threshold).float()
                            cont_mask = cont_mask * (conf_kt <=
                                                     self.threshold).float()
                        else:
                            o_update = cont_mask * 1.
                            cont_mask = cont_mask * 0.
                        if self.out_type == 'last':
                            out[:, t] = o_update[:, None] * z_kt + \
                                (1 - o_update[:, None]) * out[:, t]
                        elif self.out_type == 'conf_weighted':
                            # aggregate confidence weight
                            out_t[:, k] = z_kt
                            conf_weight = F.softmax(conf[:, t, :k], dim=-1) # (B, k)
                            z_step = out_t[:, :k] * conf_weight.unsqueeze(-1) # (B, k, C)
                            z_step = z_step.sum(dim=1) # (B, C)
                            out[:, t] = o_update[:, None] * z_step + \
                                (1 - o_update[:, None]) * out[:, t]
                        if cont_mask.sum() == 0:
                            break
                return out, conf, n_steps
        else:
            raise NotImplementedError("mode {} not implemented".format(
                self.mode))


class DACTModule(nn.Module):
    def __init__(self,
                 block_size,
                 input_size,
                 n_hidden,
                 n_layer,
                 n_class,
                 cnn_params=None,
                 pe_params=None,
                 subband_type='contiguous',
                 dropout=0,
                 max_step=None,
                 stop_by_step=False,
                 multiclass=True):
        super().__init__()
        self.block_size = block_size
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.gru = nn.GRU(input_size,
                          n_hidden,
                          n_layer,
                          batch_first=True,
                          dropout=dropout)
        self.n_class = n_class
        self.fc_h = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )
        self.multiclass = multiclass
        if multiclass:
            self.fc_out = nn.Sequential(
                nn.Linear(n_hidden, n_class),
                nn.Softmax(dim=-1),
            )
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(n_hidden, n_class),
                nn.Sigmoid(),
            )
        self.cnn = None
        if cnn_params is not None:
            self.cnn = AllConv2dModule(**cnn_params)
        self.pe = None
        if pe_params is not None:
            self.pe = PositionalEncoding(**pe_params)
        assert subband_type in ['contiguous', 'linear_gap']
        self.subband_type = subband_type
        self.max_step = max_step
        self.stop_by_step = stop_by_step

    def forward(self, x):
        '''
            x: input tensor with shape (B, F, T)
        '''
        if len(x.shape) == 4:
            assert x.shape[-1] == 1, "unsupported input channel {}".format(
                x.shape[-1])
            x = x.squeeze(-1)
        bs, n_f, n_t = x.shape
        n_sb = n_f // self.block_size  # number of subbands
        if self.subband_type == 'contiguous':
            x = x.view(bs, n_sb, self.block_size, n_t)  # (B, S, K, T)
        elif self.subband_type == 'linear_gap':
            x = x.view(bs, self.block_size, n_sb,
                       n_t).transpose(1, 2)  # (B, S, K, T)
        else:
            raise NotImplementedError("subband type {} not supported".format(
                self.subband_type))
        # chop by the step limit
        if isinstance(self.max_step, int):
            n_sb = min(self.max_step, n_sb)
            x = x[:, :n_sb]
        if self.training:
            # processing with CNNs
            if self.cnn is not None:
                x = x.reshape(bs*n_sb, -1, n_t) # (BS, K, T)
                x = self.cnn(x) # (BS, K', T)
                x = x.reshape(bs, n_sb, -1, n_t) # (B, S, K', T)
            if self.pe is not None:
                x = self.pe(x) # (B, S, K+, T)
            x = x.permute(0, 3, 1, 2).reshape(bs, -1, self.input_size) # (B, TS, K')
            rnn_out, _ = self.gru(x)
            rnn_out = rnn_out.view(bs, n_t, n_sb, self.n_hidden) # (B, T, S, D)
            hns = self.fc_h(rnn_out) # (B, T, S, 1)
            ys = self.fc_out(rnn_out) # (B, T, S, C)
            p_accum = torch.zeros(bs, n_t, 1).to(x.device) # (B, T, 1)
            p_k = hns[:, :, 0]
            out = ys[:, :, 0]
            p_accum = p_accum + p_k
            for idx in range(1, n_sb):
                out = ys[:, :, idx] * p_k + out * (1 - p_k)
                p_k = hns[:, :, idx] * p_k
                p_accum = p_accum + p_k
            return out, p_accum
        else:
            out = torch.zeros(bs, n_t,
                              self.n_class).to(x.device)  # (B, T, C)
            n_steps = torch.zeros(bs, n_t).to(x.device)  # (B, T)
            hidden = torch.zeros(self.n_layer, bs,
                                 self.n_hidden).to(x.device)
            feat_dict = {}
            for t in range(n_t):
                cont_mask = torch.ones(bs, 1).to(x.device)
                for k in range(n_sb):
                    step_inp = x[:, k] # (B, K, T)
                    # getting features if applicable
                    if self.cnn is not None:
                        # proceed only if context window greater than 1
                        if self.cnn.max_kernel_size > 1:
                            if feat_dict.get(k, None) is None:
                                step_inp = self.cnn(step_inp) # (B, K', T)
                                if self.pe is not None:
                                    step_inp = self.pe.stepwise_forward(step_inp, k) # (B, K+, T)
                                feat_dict[k] = step_inp
                            else:
                                step_inp = feat_dict[k] # (B, K', T)
                    elif self.pe is not None:
                        if feat_dict.get(k, None) is None:
                            step_inp = self.pe.stepwise_forward(step_inp, k) # (B, K+, T)
                            feat_dict[k] = step_inp
                        else:
                            step_inp = feat_dict[k] # (B, K', T)
                    # process the current step
                    step_inp = step_inp[:, :, t] # (B, K')
                    # proceed only if context window is 1
                    if self.cnn is not None and self.cnn.max_kernel_size == 1:
                        step_inp = self.cnn(step_inp.unsqueeze(-1)) # (B, K', 1)
                        if self.pe is not None:
                            step_inp = self.pe.stepwise_forward(step_inp, k)
                        step_inp = step_inp.squeeze(-1) # (B, K')
                    # recurrent processing starts here
                    s_kt, hidden_kt = self.gru(step_inp.unsqueeze(1),
                                               hidden)
                    # only update hidden states if not been processed
                    hidden = cont_mask * hidden_kt + (1 - cont_mask) * hidden  # [L, B, D]
                    h_kt = self.fc_h(s_kt).squeeze(1) # (B, 1)
                    y_kt = self.fc_out(s_kt).squeeze(1) # (B, C)
                    if k == 0:
                        o_kt = y_kt
                        p_kt = h_kt
                    else:
                        o_kt = y_kt * p_kt + o_kt * (1 - p_kt)
                        p_kt = h_kt * p_kt
                    # update step history
                    n_steps[:, t] += cont_mask.squeeze(-1).detach()
                    # conditions
                    if k < n_sb - 1:
                        # alternative stop conditions (hard step or math condition)
                        if self.stop_by_step:
                            o_update = cont_mask * 0.
                            cont_mask = cont_mask * 1.
                        else:
                            d_step = n_sb - 1 - k
                            if self.multiclass:
                                o_kt_sort = torch.sort(o_kt, dim=-1, descending=True)[0] # (B, C)
                                o_kt_sort = o_kt_sort.unsqueeze(-1) # (B, C, 1)
                                end_condition = (o_kt_sort[:, 0] * (1 - p_kt) ** d_step >= o_kt_sort[:, 1] + p_kt * d_step).float() # (B, 1)
                            else:
                                o_kt_pad = torch.stack((o_kt, 1-o_kt), dim=-1) # (B, C, 2)
                                o_kt_sort = torch.sort(o_kt_pad, dim=-1, descending=True)[0] # (B, C, 2)
                                end_condition = (o_kt_sort[:, :, 0] * (1 - p_kt)**d_step >= o_kt_sort[:, :, 1] + p_kt * d_step).float() # (B, C)
                                end_condition = (end_condition.sum(dim=-1, keepdim=True) >= self.n_class).float() # (B, 1)
                            o_update = cont_mask * end_condition
                            cont_mask = cont_mask * (1 - end_condition)
                    else:
                        o_update = cont_mask * 1.
                        cont_mask = cont_mask * 0.
                    out[:, t] = o_update * o_kt + (1 - o_update) * out[:, t]
                    if cont_mask.sum() == 0:
                        break
            return out, n_steps


class DACTCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layer,
                 n_class,
                 cnn_params=None,
                 pe_params=None,
                 dropout=0,
                 max_step=None,
                 stop_by_step=False,
                 multiclass=True,
                 ):
        super(DACTCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.max_step = max_step
        self.cell_layers = nn.ModuleList()
        for i in range(n_layer):
            self.cell_layers.append(
                nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size))
        self.n_layer = n_layer
        self.fc_h = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.multiclass = multiclass
        if multiclass:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_size, n_class),
                nn.Softmax(dim=-1),
            )
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_size, n_class),
                nn.Sigmoid(),
            )
        self.stop_by_step = stop_by_step
        self.cnn = None
        if cnn_params is not None:
            self.cnn = AllConv2dModule(**cnn_params)
        self.pe = None
        if pe_params is not None:
            self.pe = PositionalEncoding(**pe_params)

    def forward(self, inp, feat_dict, t_idx, hidden=None):
        '''
            inp: tensor of shape (B,S,K,T)
            inp_feat: dictionary of pairs step:feat
            t_idx: temporal index

            returns
                y_accum: (B, C)
        '''
        bs = inp.shape[0]
        n_band = inp.shape[1]
        s_accum = torch.zeros(
            (self.n_layer, bs, self.hidden_size)).to(inp.device)  # s_t
        y_accum = torch.zeros((bs, self.n_class)).to(inp.device)  # y_t
        if isinstance(self.max_step, int):
            max_step = min(n_band, self.max_step)
        else:
            max_step = n_band
        if hidden is None:
            hidden = torch.zeros_like(s_accum) # (L, B, D)
        if self.training:
            for step in range(max_step):
                step_inp = inp[:, step] # (B, K, T)
                # getting features if applicable
                if self.cnn is not None:
                    if feat_dict.get(step, None) is None:
                        step_inp = self.cnn(step_inp) # (B, K' T)
                        if self.pe is not None:
                            step_inp = self.pe.stepwise_forward(step_inp, step) # (B, K+, T)
                        feat_dict[step] = step_inp
                    else:
                        step_inp = feat_dict[step] # (B, K+, T)
                elif self.pe is not None:
                    if feat_dict.get(step, None) is None:
                        step_inp = self.pe.stepwise_forward(step_inp, step) # (B, K+, T)
                    else:
                        step_inp = feat_dict[step] # (B, K+, T)
                # process the current step
                step_inp = step_inp[:, :, t_idx] # (B, K')
                # recurrent processing starts here
                tmp_hidden = torch.zeros_like(hidden)
                for layer_idx, cell in enumerate(self.cell_layers):
                    step_inp = cell(step_inp, hidden[layer_idx])
                    tmp_hidden[layer_idx] = step_inp
                hidden = tmp_hidden
                h_step = self.fc_h(hidden[self.n_layer - 1]) # (B, 1)
                y_step = self.fc_out(hidden[self.n_layer - 1]) # (B, C)
                p_accum = torch.zeros(bs, 1).to(inp.device) # keep track of ponder loss
                if step == 0:
                    y_accum = y_step
                    p_step = h_step
                    s_accum = hidden
                else:
                    y_accum = y_step * p_step + y_accum * (1 - p_step)
                    p_step = p_step * h_step
                    s_accum = hidden * p_step[None, :] + s_accum * (1 - p_step)[None, :]
                p_accum = p_accum + p_step
            return y_accum, s_accum, p_accum
        else:
            cont_mask = torch.ones(bs, 1).to(inp.device)
            out_t = torch.zeros_like(y_accum) # (B, C), returned as output
            hidden_t = torch.zeros_like(s_accum)
            steps_t = torch.zeros(bs, 1).to(inp.device)  # (B, 1)
            for step in range(max_step):
                step_inp = inp[:, step] # (B, K, T)
                # getting features if applicable
                if self.cnn is not None:
                    # proceed only if context window greater than 1
                    if self.cnn.max_kernel_size > 1:
                        if feat_dict.get(step, None) is None:
                            step_inp = self.cnn(step_inp) # (B, K' T)
                            if self.pe is not None:
                                step_inp = self.pe.stepwise_forward(step_inp, step) # (B, K+, T)
                            feat_dict[step] = step_inp
                        else:
                            step_inp = feat_dict[step] # (B, K+, T)
                elif self.pe is not None:
                    if feat_dict.get(step, None) is None:
                        step_inp = self.pe.stepwise_forward(step_inp, step) # (B, K+, T)
                    else:
                        step_inp = feat_dict[step] # (B, K+, T)
                # process the current step
                step_inp = step_inp[:, :, t_idx] # (B, K')
                # proceed only if context window is 1
                if self.cnn is not None and self.cnn.max_kernel_size == 1:
                    step_inp = self.cnn(step_inp.unsqueeze(-1)) # (B, K', 1)
                    if self.pe is not None:
                        step_inp = self.pe.stepwise_forward(step_inp, step)
                    step_inp = step_inp.squeeze(-1) # (B, K')
                # recurrent processing starts here
                tmp_hidden = torch.zeros_like(hidden)
                for layer_idx, cell in enumerate(self.cell_layers):
                    step_inp = cell(step_inp, hidden[layer_idx])
                    tmp_hidden[layer_idx] = step_inp
                hidden = tmp_hidden
                h_step = self.fc_h(hidden[self.n_layer - 1]) # (B, 1)
                y_step = self.fc_out(hidden[self.n_layer - 1]) # (B, C)
                steps_t += cont_mask.detach()
                if step == 0:
                    y_accum = y_step
                    p_step = h_step
                    s_accum = hidden
                else:
                    y_accum = y_step * p_step + y_accum * (1 - p_step)
                    p_step = p_step * h_step
                    s_accum = hidden * p_step[None, :] + s_accum * (1 - p_step)[None, :]
                '''
                    end condition follows DACT
                '''
                if step < max_step - 1:
                    if self.stop_by_step:
                        o_update = cont_mask * 0.
                        cont_mask = cont_mask * 1.
                    else:
                        d_step = max_step - 1 - step
                        if self.multiclass:
                            out_sort = torch.sort(y_accum, dim=-1, descending=True)[0]
                            out_sort = out_sort.unsqueeze(-1) # (B, C, 1)
                            end_mask = (out_sort[:, 0] * (1 - p_step) ** d_step >= out_sort[:, 1] + p_step * d_step).float()
                        else:
                            out_pad = torch.stack((y_accum, 1-y_accum), -1) # (B, C, 2)
                            out_sort = torch.sort(out_pad, dim=-1, descending=True)[0] # (B, C, 2)
                            end_mask = (out_sort[:, :, 0] * (1 - p_step) ** d_step >= out_sort[:, :, 1] + p_step * d_step).float() # (B, C)
                            end_mask = (end_mask.sum(dim=-1, keepdim=True) == self.n_class).float() # (B, 1)

                        o_update = cont_mask * end_mask
                        cont_mask = cont_mask * (1 - end_mask)
                else:
                    o_update = cont_mask * 1.
                    cont_mask = cont_mask * 0.
                out_t = o_update * y_accum + (1 - o_update) * out_t
                hidden_t = o_update[None, :] * s_accum + (1 - o_update)[None, :] * hidden_t
                if cont_mask.sum() == 0:
                    break
            return out_t, hidden_t, steps_t


class DACTBlock(nn.Module):
    def __init__(self, cell_params):
        super(DACTBlock, self).__init__()
        self.n_class = cell_params['n_class']
        self.block_size = cell_params.pop('block_size')
        self.subband_type = cell_params.pop('subband_type')
        assert self.subband_type in ['contiguous', 'linear_gap']
        self.cell = DACTCell(**cell_params)

    def forward(self, inp, hidden=None):
        '''
            inp: tensor with shape [B, F, L]

            returns:
                out: [B, L, C]
                ponder_cost: [B, L, 1]
        '''
        if len(inp.shape) == 4:
            assert inp.shape[-1] == 1, "unsupported input channel {}".format(
                inp.shape[-1])
            inp = inp.squeeze(-1)
        bs = inp.shape[0]
        n_freq = inp.shape[1]
        n_band = n_freq // self.block_size
        seq_len = inp.shape[-1]
        # subband partition
        if self.subband_type == 'contiguous':
            inp = inp.view(bs, n_band, self.block_size, seq_len)  # (B, S, K, L)
        elif self.subband_type == 'linear_gap':
            inp = inp.view(bs, self.block_size,
                           n_band, seq_len).transpose(1, 2)  # (B, S, K, L)
        else:
            raise NotImplementedError("subband type {} not supported".format(
                self.subband_type))
        out = torch.zeros(seq_len, bs,
                          self.n_class).to(inp.device)  # [L, B, C]
        inp_feat_dict = {}
        if self.training:
            p_cost = torch.zeros(seq_len, bs, 1).to(inp.device)
            for t in range(seq_len):
                y_t, hidden, p_t = self.cell(inp, inp_feat_dict, t, hidden)
                out[t] = y_t
                p_cost[t] = p_t
            out = out.transpose(0, 1) # (B, L, C)
            p_cost = p_cost.transpose(0, 1)
            return out, p_cost
        else:
            n_cost = torch.zeros(seq_len, bs, 1).to(inp.device)
            for t in range(seq_len):
                y_t, hidden, steps_t = self.cell(inp, inp_feat_dict, t, hidden)
                out[t] = y_t
                n_cost[t] = steps_t
            out = out.transpose(0, 1) # (B, L, C)
            n_cost = n_cost.transpose(0, 1)
            return out, n_cost


class ACTCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layer,
                 n_class,
                 ponder_eps,
                 max_step,
                 cnn_params=None,
                 pe_params=None,
                 init_halt_bias=0,
                 multiclass=True):
        super(ACTCell, self).__init__()
        if ponder_eps < 0 or ponder_eps > 1:
            raise ValueError("ponder epsilon must be between 0 and 1")
        self.input_size = input_size
        self.ponder_eps = ponder_eps
        self.max_step = max_step
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.cell_layers = nn.ModuleList()
        for i in range(n_layer):
            self.cell_layers.append(
                nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size))
        self.n_layer = n_layer
        self.halt_fn = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        if init_halt_bias is not None:
            self.halt_fn[0].bias.data.fill_(init_halt_bias)
        self.multiclass = multiclass
        if multiclass:
            self.out = nn.Sequential(nn.Linear(hidden_size, n_class),
                                     nn.Softmax(dim=-1))
        else:
            self.out = nn.Sequential(nn.Linear(hidden_size, n_class),
                                     nn.Sigmoid())
        self.cnn = None
        if cnn_params is not None:
            self.cnn = AllConv2dModule(**cnn_params)
        self.pe = None
        if pe_params is not None:
            self.pe = PositionalEncoding(**pe_params)

    def forward(self, inp, feat_dict, t_idx, hidden=None):
        '''
            inp: tensor of shape (B,S,K,T)
            inp_feat: dictionary of pairs step:feat
            t_idx: temporal index

            returns
                y_accum: (B, C)
                s_accum: (B, S, K)
                P_accum: (B, 1)
        '''
        bs = inp.shape[0]
        n_band = inp.shape[1]
        budget = torch.ones((bs, 1)).to(inp.device) - self.ponder_eps
        s_accum = torch.zeros(
            (self.n_layer, bs, self.hidden_size)).to(inp.device)  # s_t
        y_accum = torch.zeros((bs, self.n_class)).to(inp.device)  # y_t
        N_accum = torch.ones_like(budget)  # N(t)
        R_accum = torch.zeros_like(budget)  # R(t)
        halt_accum = torch.zeros_like(budget)
        cont_mask = torch.ones_like(budget)
        if isinstance(self.max_step, int):
            max_step = min(n_band, self.max_step)
        else:
            max_step = n_band
        if hidden is None:
            hidden = torch.zeros_like(s_accum)
        for step in range(max_step):
            step_inp = inp[:, step] # (B, K, T)
            # getting features if applicable
            if self.cnn is not None:
                # proceed only if context window greater than 1
                if self.training or self.cnn.max_kernel_size > 1:
                    if feat_dict.get(step, None) is None:
                        step_inp = self.cnn(step_inp) # (B, K' T)
                        if self.pe is not None:
                            step_inp = self.pe.stepwise_forward(step_inp, step) # (B, K+, T)
                        feat_dict[step] = step_inp
                    else:
                        step_inp = feat_dict[step] # (B, K+, T)
            elif self.pe is not None:
                if feat_dict.get(step, None) is None:
                    step_inp = self.pe.stepwise_forward(step_inp, step) # (B, K+, T)
                else:
                    step_inp = feat_dict[step] # (B, K+, T)
            # process the current step
            step_inp = step_inp[:, :, t_idx] # (B, K')
            # proceed only if context window is 1
            if self.cnn is not None and not self.training and self.cnn.max_kernel_size == 1:
                step_inp = self.cnn(step_inp.unsqueeze(-1)) # (B, K', 1)
                if self.pe is not None:
                    step_inp = self.pe.stepwise_forward(step_inp, step)
                step_inp = step_inp.squeeze(-1) # (B, K')
            # recurrent processing starts here
            tmp_hidden = torch.zeros_like(hidden)
            for layer_idx, cell in enumerate(self.cell_layers):
                step_inp = cell(step_inp, hidden[layer_idx])
                tmp_hidden[layer_idx] = step_inp
            hidden = tmp_hidden
            step_y = self.out(hidden[self.n_layer - 1])
            step_halt = self.halt_fn(hidden[self.n_layer - 1])
            '''
                1) reaches N(t), end at current step (end_mask)
                2) continue (cont_mask)
                3) already terminated
            '''
            if step == max_step - 1:
                end_mask = cont_mask * 1.
                cont_mask = cont_mask * 0.
            else:
                end_mask = cont_mask * (step_halt + halt_accum >=
                                        budget).float()
                cont_mask = cont_mask * (step_halt + halt_accum <
                                         budget).float()
            masked_halt = step_halt * cont_mask
            masked_rmd = end_mask * (1 - halt_accum)
            p_step = masked_halt + masked_rmd  # p_tn
            N_accum += cont_mask.detach()
            R_accum += masked_rmd
            s_accum += p_step * hidden
            y_accum += p_step * step_y
            halt_accum += p_step
            if cont_mask.sum() == 0:
                break
        return y_accum, s_accum, N_accum, R_accum


class ACTBlock(nn.Module):
    def __init__(self, cell_params):
        super(ACTBlock, self).__init__()
        self.n_class = cell_params['n_class']
        self.block_size = cell_params.pop('block_size')
        self.subband_type = cell_params.pop('subband_type')
        assert self.subband_type in ['contiguous', 'linear_gap']
        self.cell = ACTCell(**cell_params)

    def forward(self, inp, hidden=None):
        '''
            inp: tensor with shape [B, F, L]

            returns:
                out: [B, L, C]
                ponder_cost: [B, 1]
        '''
        if len(inp.shape) == 4:
            assert inp.shape[-1] == 1, "unsupported input channel {}".format(
                inp.shape[-1])
            inp = inp.squeeze(-1)
        bs = inp.shape[0]
        n_freq = inp.shape[1]
        n_band = n_freq // self.block_size
        seq_len = inp.shape[-1]
        # subband partition
        if self.subband_type == 'contiguous':
            inp = inp.view(bs, n_band, self.block_size, seq_len)  # (B, S, K, L)
        elif self.subband_type == 'linear_gap':
            inp = inp.view(bs, self.block_size,
                           n_band, seq_len).transpose(1, 2)  # (B, S, K, L)
        else:
            raise NotImplementedError("subband type {} not supported".format(
                self.subband_type))
        out = torch.zeros(seq_len, bs,
                          self.n_class).to(inp.device)  # [L, B, C]
        N_cost = torch.zeros(bs, 1).to(inp.device)
        R_cost = torch.zeros_like(N_cost)
        inp_feat_dict = {}
        for t in range(seq_len):
            y_t, hidden, N_t, R_t = self.cell(inp, inp_feat_dict, t, hidden)
            out[t] = y_t
            N_cost += N_t
            R_cost += R_t
        out = out.permute(1, 0, 2)
        # average across frames
        N_cost = N_cost / seq_len
        R_cost = R_cost / seq_len
        return out, N_cost, R_cost


def test_ACT():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=12, size=(bs, seq)).to(device)

    cell_params = {
        'block_size': 16,
        'input_size': 144,
        'hidden_size': 128,
        'n_layer': 3,
        'n_class': 12,
        'cnn_params': {
            'channels': (16, 32, 64, 128, 128, 128, 128),
            'kernel_sizes': 1,
            'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
            'dropout': 0.5,
        },
        #  'pe_params': None,
        'pe_params': {
            'd_model': 16,
            'max_len': 8,
            'dropout': 0.1,
            'trainable': False,
            'mode': 'cat',
        },
        'subband_type': 'contiguous',
        'ponder_eps': 1e-2,
        'max_step': 128,
        'init_halt_bias': None,
    }
    net = ACTBlock(cell_params).to(device)
    out, N_cost, R_cost = net(x)
    loss = nn.NLLLoss()(out.permute(0, 2, 1), y)
    loss.backward()


def test_ACT_multilabel():
    bs = 16
    seq = 100
    feat = 128
    n_class = 11
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=2, size=(bs, seq, n_class)).to(device).float()

    cell_params = {
        'block_size': 16,
        'input_size': 128,
        'hidden_size': 128,
        'n_layer': 3,
        'n_class': n_class,
        'cnn_params': {
            'channels': (16, 32, 64, 128, 128, 128, 128),
            'kernel_sizes': 3,
            'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
            'dropout': 0.5,
        },
        'pe_params': None,
        'subband_type': 'contiguous',
        'ponder_eps': 1e-2,
        'max_step': 128,
        'init_halt_bias': None,
        'multiclass': False,
    }
    net = ACTBlock(cell_params).to(device)
    out, N_cost, R_cost = net(x)
    loss = nn.BCELoss()(out, y)
    loss.backward()


def test_fublock():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=12, size=(bs, seq)).to(device)

    block_size = 16
    input_size = 128
    n_hidden = 128
    n_layer = 1
    n_class = 12
    subband_type = 'contiguous'
    dropout = 0.
    threshold = 1.0
    max_step = None
    rand_step = True
    mode = 'l2_naive'
    out_type = 'conf_weighted'
    #  cnn_params = None
    cnn_params = {
        'channels': (16, 32, 64, 128, 128, 128, 128),
        'kernel_sizes': 1,
        'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
        'dropout': 0.5,
    }
    pe_params = None
    #  pe_params = {
    #      'd_model': 16,
    #      'max_len': 8,
    #      'dropout': 0.1,
    #      'trainable': False,
    #      'mode': 'cat',
    #  }
    net = FreqUnrollBlock(
        block_size,
        input_size,
        n_hidden,
        n_layer,
        n_class,
        cnn_params,
        pe_params,
        subband_type,
        dropout,
        threshold,
        max_step,
        rand_step,
        mode,
        out_type
    ).to(device)
    pdb.set_trace()
    out, _ = net(x)
    loss = nn.NLLLoss()(out.permute(0, 2, 1), y)
    loss.backward()
    net.eval()
    out = net(torch.randn_like(x))


def test_fublock_multilabel():
    bs = 16
    seq = 100
    feat = 128
    n_class = 11
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=2, size=(bs, seq, n_class)).to(device).float()

    block_size = 16
    input_size = 128
    n_hidden = 512
    n_layer = 1
    subband_type = 'contiguous'
    dropout = 0.
    threshold = 1.0
    max_step = None
    rand_step = False
    mode = 'l2_naive'
    out_type = 'last'
    multiclass = False
    cnn_params = {
        'channels': (16, 32, 64, 128, 128, 128, 128),
        'kernel_sizes': 3,
        'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
        'dropout': 0.5,
    }
    pe_params = None
    net = FreqUnrollBlock(
        block_size,
        input_size,
        n_hidden,
        n_layer,
        n_class,
        cnn_params,
        pe_params,
        subband_type,
        dropout,
        threshold,
        max_step,
        rand_step,
        mode,
        out_type,
        multiclass
    ).to(device)
    pdb.set_trace()
    out, _ = net(x)
    loss = nn.BCELoss()(out, y)
    loss.backward()
    net.eval()
    out = net(torch.randn_like(x))


def test_dact():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=12, size=(bs, seq)).to(device)

    block_size = 16
    input_size = 144
    n_hidden = 512
    n_layer = 1
    n_class = 12
    #  cnn_params = None
    cnn_params = {
        'channels': (16, 32, 64, 128, 128, 128, 128),
        'kernel_sizes': 1,
        'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
        'dropout': 0.5,
    }
    #  pe_params = None
    pe_params = {
        'd_model': 16,
        'max_len': 8,
        'dropout': 0.1,
        'trainable': False,
        'mode': 'cat',
    }
    subband_type = 'contiguous'
    dropout = 0.2
    max_step = 4
    stop_by_step = False
    net = DACTModule(
        block_size,
        input_size,
        n_hidden,
        n_layer,
        n_class,
        cnn_params,
        pe_params,
        subband_type,
        dropout,
        max_step,
        stop_by_step
    ).to(device)
    out, p_cost = net(x)
    loss = nn.NLLLoss()(out.permute(0, 2, 1), y)
    loss.backward()
    pdb.set_trace()
    net.eval()
    out = net(torch.randn_like(x))


def test_dact_multilabel():
    bs = 16
    seq = 100
    feat = 128
    n_class = 11
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=2, size=(bs, seq, n_class)).to(device).float()

    block_size = 16
    input_size = 128
    n_hidden = 512
    n_layer = 1
    cnn_params = {
        'channels': (16, 32, 64, 128, 128, 128, 128),
        'kernel_sizes': 3,
        'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
        'dropout': 0.5,
    }
    pe_params = None
    subband_type = 'contiguous'
    dropout = 0.2
    max_step = 4
    stop_by_step = False
    multiclass = False
    net = DACTModule(
        block_size,
        input_size,
        n_hidden,
        n_layer,
        n_class,
        cnn_params,
        pe_params,
        subband_type,
        dropout,
        max_step,
        stop_by_step,
        multiclass
    ).to(device)
    out, p_cost = net(x)
    loss = nn.BCELoss()(out, y)
    loss.backward()
    pdb.set_trace()
    net.eval()
    out = net(torch.randn_like(x))


def test_our_DACT():
    bs = 16
    seq = 100
    feat = 128
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=12, size=(bs, seq)).to(device)

    cell_params = {
        'block_size': 16,
        'input_size': 144,
        'hidden_size': 128,
        'n_layer': 1,
        'n_class': 12,
        #  'cnn_params': None,
        'cnn_params': {
            'channels': (16, 32, 64, 128, 128, 128, 128),
            'kernel_sizes': 1,
            'pooling_sizes': [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
            'dropout': 0.5,
        },
        #  'pe_params': None,
        'pe_params': {
            'd_model': 16,
            'max_len': 8,
            'dropout': 0.1,
            'trainable': False,
            'mode': 'cat',
        },
        'subband_type': 'contiguous',
        'dropout': 0,
        'max_step': None,
        'stop_by_step': False,
        'multiclass': True,
    }
    net = DACTBlock(cell_params).to(device)
    out, p_cost = net(x)
    loss = nn.NLLLoss()(out.permute(0, 2, 1), y)
    loss.backward()
    pdb.set_trace()
    net.eval()
    out, n_cost = net(torch.randn_like(x))


def test_our_DACT_multilabel():
    bs = 16
    seq = 100
    feat = 128
    n_class = 11
    device = torch.device('cuda:0')
    x = torch.randn(bs, feat, seq).to(device)
    y = torch.randint(low=0, high=2, size=(bs, seq, n_class)).to(device).float()

    cell_params = {
        'block_size': 16,
        'input_size': 16,
        'hidden_size': 128,
        'n_layer': 1,
        'n_class': n_class,
        'cnn_params': None,
        'pe_params': None,
        'subband_type': 'contiguous',
        'dropout': 0,
        'max_step': None,
        'stop_by_step': False,
        'multiclass': False,
    }
    net = DACTBlock(cell_params).to(device)
    out, p_cost = net(x)
    loss = nn.BCELoss()(out, y)
    loss.backward()
    net.eval()
    out, n_cost = net(torch.randn_like(x))


if __name__ == '__main__':
    test_fublock()
