# An (almost faithful) port of
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
import jax
import jax.numpy as jnp
import haiku as hk


class Transformer(hk.Module):
    def __init__(self, d_model, nheads, encoder_layers=6, decoder_layers=6,
            dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(encoder_layers, layer_args=dict(
            d_model=d_model, nheads=nheads,
            dim_feedforward=dim_feedforward, dropout=dropout
        ))
        self.decoder = TransformerDecoder(decoder_layers, layer_args=dict(
            d_model=d_model, nheads=nheads,
            dim_feedforward=dim_feedforward, dropout=dropout
        ))

    def __call__(self, source, target, is_training):
        source = self.encoder(source, is_training)
        return self.decoder(target, source, is_training)


class TransformerEncoder(hk.Module):
    def __init__(self, num_layers, layer_args):
        super().__init__()
        self.num_layers = num_layers
        self.layer_args = layer_args

    def __call__(self, x, is_training):
        """Params:
        x: B x T x C
        """
        for _ in range(self.num_layers):
            x = EncoderLayer(**self.layer_args)(x, is_training)
        return x


class TransformerDecoder(hk.Module):
    def __init__(self, num_layers, layer_args):
        super().__init__()
        self.num_layers = num_layers
        self.layer_args = layer_args

    def __call__(self, target, source, is_training):
        """Params:
        x: B x T x C
        """
        x = target
        for _ in range(self.num_layers):
            x = DecoderLayer(**self.layer_args)(x, source, is_training)
        return x


class EncoderLayer(hk.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.self_attn = hk.MultiHeadAttention(nheads, d_model//nheads, 1.0)
        self.linear1 = hk.Linear(dim_feedforward)
        self.linear2 = hk.Linear(d_model)

        self.norm1 = hk.LayerNorm(-1, True, True)
        self.norm2 = hk.LayerNorm(-1, True, True)

    def __call__(self, x, is_training):
        """Params:
        x: B x T x C
        """
        # Self-Attention Block
        resid = x
        x = self.self_attn(x, x, x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm1(x)
        x = resid + x

        # Feedforward-Block
        resid = x
        x = jax.nn.relu(self.linear1(x))
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.linear2(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm2(x)
        x = resid + x

        return x


class DecoderLayer(hk.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.self_attn  = hk.MultiHeadAttention(nheads, d_model//nheads, 1.0)
        self.cross_attn = hk.MultiHeadAttention(nheads, d_model//nheads, 1.0)
        self.linear1 = hk.Linear(dim_feedforward)
        self.linear2 = hk.Linear(d_model)

        self.norm1 = hk.LayerNorm(-1, True, True)
        self.norm2 = hk.LayerNorm(-1, True, True)
        self.norm3 = hk.LayerNorm(-1, True, True)

    def __call__(self, target, source, is_training):
        """Params:
        target: B x T x C
        source: B x N x C
        """

        # Self-Attention Block
        x = target
        resid = x
        x = self.self_attn(x, x, x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm1(x)
        x = resid + x

        # Cross-Attention Block
        resid = x
        x = self.cross_attn(x, source, source)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm2(x)
        x = resid + x

        # Feedforward-Block
        resid = x
        x = jax.nn.relu(self.linear1(x))
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.linear2(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.norm3(x)
        x = resid + x

        return x

