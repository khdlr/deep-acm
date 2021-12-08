import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk

from .. import nnutils as nn


class Xception(hk.Module):
    """Xception backbone like the one used in CALFIN"""
    def __call__(self, x, is_training=False):
        B, H, W, C = x.shape

        # Backbone
        x, skip1 = XceptionBlock([128, 128, 128], stride=2, return_skip=True)(x, is_training)
        x, skip2 = XceptionBlock([256, 256, 256], stride=2, return_skip=True)(x, is_training)
        x, skip3 = XceptionBlock([768, 768, 768], stride=2, return_skip=True)(x, is_training)
        for i in range(8):
            x = XceptionBlock([768, 768, 768], skip_type='sum', stride=1)(x, is_training)

        x = XceptionBlock([ 728, 1024, 1024], stride=2)(x, is_training)
        x = XceptionBlock([1536, 1536, 2048], stride=1, rate=(1, 2, 4))(x, is_training)

        # ASPP
        # Image Feature branch
        bD = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
        bD = nn.ConvBNAct(256, 1, act='elu')(bD, is_training)
        bD = nn.upsample(bD, factor=2)

        b0 = nn.ConvBNAct(256, 1, act='elu')(x, is_training)
        b1 = nn.SepConvBN(256, rate=1)(x, is_training)
        b2 = nn.SepConvBN(256, rate=2)(x, is_training)
        b3 = nn.SepConvBN(256, rate=3)(x, is_training)
        b4 = nn.SepConvBN(256, rate=4)(x, is_training)
        b5 = nn.SepConvBN(256, rate=5)(x, is_training)
        x = jnp.concatenate([bD, b0, b1, b2, b3, b4, b5], axis=-1)

        x = nn.ConvBNAct(256, 1, act='elu')(x, is_training)
        if is_training:
            x = nn.channel_dropout(x, 0.2)

        skip3 = nn.ConvBNAct(48, 1, act='elu')(skip3, is_training)
        return [skip3, x]


class XceptionSlim(hk.Module):
    """Xception backbone like the one used in CALFIN"""
    def __call__(self, x, is_training=False):
        B, H, W, C = x.shape

        # Backbone
        x, skip1 = XceptionBlock([ 32,  32,  32], stride=2, return_skip=True)(x, is_training)
        x, skip2 = XceptionBlock([ 64,  64,  64], stride=2, return_skip=True)(x, is_training)
        x, skip3 = XceptionBlock([128, 128, 128], stride=2, return_skip=True)(x, is_training)
        for i in range(8):
            x = XceptionBlock([256, 256, 256], skip_type='sum', stride=1)(x, is_training)

        x = XceptionBlock([256, 256, 256], stride=2)(x, is_training)
        x = XceptionBlock([512, 512, 512], stride=1, rate=(1, 2, 4))(x, is_training)

        # ASPP
        # Image Feature branch
        bD = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
        bD = nn.ConvBNAct(64, 1, act='elu')(bD, is_training)
        bD = nn.upsample(bD, factor=2)

        b0 = nn.ConvBNAct(64, 1, act='elu')(x, is_training)
        b1 = nn.SepConvBN(64, rate=1)(x, is_training)
        b2 = nn.SepConvBN(64, rate=2)(x, is_training)
        b3 = nn.SepConvBN(64, rate=4)(x, is_training)
        b4 = nn.SepConvBN(64, rate=8)(x, is_training)
        x = jnp.concatenate([bD, b0, b1, b2, b4], axis=-1)

        x = nn.ConvBNAct(256, 1, act='elu')(x, is_training)
        if is_training:
            x = nn.channel_dropout(x, 0.5)

        return [skip2, x]


class XceptionBlock(hk.Module):
    def __init__(self, depth_list, stride, skip_type='conv',
                 rate=1, return_skip=False):
        super().__init__()
        self.blocks = []
        if rate == 1:
            rate = [1, 1, 1]
        for i in range(3):
            self.blocks.append(nn.SepConvBN(
                depth_list[i],
                stride=stride if i == 2 else 1,
                rate=rate[i],
            ))

        if skip_type == 'conv':
            self.shortcut = nn.ConvBNAct(depth_list[-1], 1, stride=stride, act=None)
        elif skip_type == 'sum':
            self.shortcut = nn.identity
        self.return_skip = return_skip

    def __call__(self, inputs, is_training):
        residual = inputs
        for i, block in enumerate(self.blocks):
            residual = block(residual, is_training)
            if i == 1:
                skip = residual

        shortcut = self.shortcut(inputs, is_training)
        outputs = residual + shortcut

        if self.return_skip:
            return outputs, skip
        else:
            return outputs

