import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer_enc_dec import Encoder, EncoderLayer
from layers.svs import SVS
from layers.attention import FullAttention, AttentionLayer
from layers.embed import FeaturesEmbedding
import numpy as np


class Model(nn.Module):
    """
    Transplit model
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.svs = SVS(
            configs.enc_in + len(configs.float_features),
            configs.d_model,
            configs.c_out,
            configs.period,
            configs.n_filters
        )
        self.enc_embedding = FeaturesEmbedding(
            configs.d_model,
            configs.categorical_features,
        ).to("cuda")

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.output = nn.Linear(configs.d_model, configs.n_filters)


    def forward(
            self,
            x_enc: torch.Tensor,
            x_mark_enc: torch.Tensor,
            x_dec: torch.Tensor,
            x_mark_dec: torch.Tensor,
            enc_self_mask=None
        ) -> torch.Tensor:
        
        x = self.svs.encode(x_enc)
        x_mark_enc = x_mark_enc[:, ::self.svs.period, :]

        enc_out = self.enc_embedding(x, x_mark_enc) # shape (batch_size, seq_len // period, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.output(enc_out)
        pred_days = (self.pred_len - 1) // self.svs.period + 1
        dec_out = dec_out[:, :pred_days, :]
        dec_out = self.svs.decode(dec_out)

        return dec_out[:, :self.pred_len, :]


# python run.py --data_path datasets/creos.csv --model Transplit --batch_size 32 --external_factors datasets/external_factors.csv