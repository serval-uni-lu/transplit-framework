import torch
import torch.nn as nn
import numpy as np
from layers.embed import DataEmbedding
from layers.diffusion import unet
from layers.diffusion import torch_dct
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.n_channels = configs.d_model // 8
        self.num_steps = configs.num_steps
        self.batch_size = configs.batch_size
        self.training_noise = configs.training_noise
        self.sampling_noise = configs.sampling_noise
        self.device = device

        self.blur_sigmas = torch.tensor(np.exp(
            np.linspace(np.log(configs.blur_sigma_min), np.log(configs.blur_sigma_max), configs.num_steps)
        )).float().to(device)
        freqs = np.pi * torch.range(0, self.pred_len - 1) / self.pred_len
        self.frequencies_squared = (freqs**2).float().to(device)

        self.embedding = DataEmbedding(
            # configs.enc_in,
            1,
            self.n_channels,
            configs.embed,
            configs.freq,
            configs.dropout
        ).float()

        self.unet = unet.UNet(
            input_channels = self.n_channels,
            context_channels = self.n_channels,
            output_channels = 1,
            n_channels = self.n_channels,
            ch_mults = (1, 2, 2, 2),
            is_attn = (False, True, True, True),
            causal = configs.use_causal,
            n_blocks = 2
        )
    
    def get_t(self, i: int = None) -> torch.Tensor:
        if i is None:
            i_min = self.num_steps // 4
            i_max = self.num_steps * 3 // 4
            t = torch.randint(i_min, i_max, (self.batch_size,))
        else:
            t = torch.full((self.batch_size,), i, dtype=torch.int64)
        return t.to(self.device)
    
    def blur(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, pred_len, n_channels)
        # t: (batch_size,)
        sigmas = self.blur_sigmas[t][:, None, None]
        sigmas = sigmas**2 / 2
        x = x.transpose(1, 2)
        dct_coefs = torch_dct.dct(x, norm='ortho')
        # exp = torch.exp(-self.frequencies_squared * sigmas)
        dct_coefs = dct_coefs * torch.exp(-self.frequencies_squared * sigmas)
        x_t = torch_dct.idct(dct_coefs, norm='ortho')
        x_t = x_t.transpose(1, 2)
        return x_t

    def q_xt_x0(self, batch_y, t):
        x_t = self.blur(batch_y, t)
        x_t = x_t + self.training_noise * torch.randn_like(batch_y, device=batch_y.device)
        x_t_1 = self.blur(batch_y, t - 1)
        eps = x_t_1 - x_t
        return x_t, eps

    def p_xt(self, xt: torch.Tensor, eps_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(xt) * (t[:, None, None] > 0).float() # remove noise if we go to t=0
        xt = xt + eps_hat
        xt = xt + noise * self.sampling_noise # * self.blur_sigmas[t][:, None, None]
        return xt

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, t):
        if x_enc is not None:
            context = self.embedding(x_enc, x_mark_enc)
        else:
            context = None
        to_deblur = self.embedding(x_dec, x_mark_dec)
        x = self.unet(to_deblur, t, context)
        return x
    
    def iterate_through_inference(self, x_enc, x_mark_enc, x_dec, x_mark_dec, steps=None, factor=5.0):
        if steps is None:
            steps = self.num_steps
        yield x_dec
        for i in tqdm(range(steps-1, -1, -1)):
            t = self.get_t(i)
            pred_free = self.forward(None, None, x_dec, x_mark_dec, t)
            pred_with_context = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, t)
            pred = pred_free + (pred_with_context - pred_free) * factor
            x_dec = self.p_xt(x_dec, pred, t)
            yield x_dec
    
    def inference(self, x_enc, x_mark_enc, x_dec, x_mark_dec, steps=None, factor=5.0):
        for x in self.iterate_through_inference(x_enc, x_mark_enc, x_dec, x_mark_dec, steps, factor):
            pass
        return x


if __name__ == "__main__":
    class Configs:
        pass
    configs = Configs()
    configs.d_model = 64
    configs.pred_len = 168
    configs.num_steps = 50
    configs.blur_sigma_min = 0.5
    configs.blur_sigma_max = 16
    configs.n_channels = 8
    configs.enc_in = 1
    configs.embed = 'timeF'
    configs.freq = "h"
    configs.dropout = 0.05
    model = Model(configs)
    batch_size = 32
    pred_length = 168
    x_enc = torch.randn(batch_size, pred_length, 1)
    x_mark_enc = torch.zeros(batch_size, pred_length, 1)
    x_dec = torch.randn(batch_size, pred_length, 1)
    x_mark_dec = torch.zeros(batch_size, pred_length, 1)
    t = torch.randint(0, 100, (batch_size, pred_length))
    y = model(x_enc, x_mark_enc, x_dec, x_mark_dec, t)
    print(y.shape)


# python run.py --data_path datasets/creos.csv --model Dedipeak --batch_size 64