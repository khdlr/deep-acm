import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange
from jax.experimental.host_callback import id_print

init = hk.initializers.Orthogonal()
act = jax.nn.leaky_relu


class ResBlock(hk.Module):
  def __init__(self, channels, kernel_size=3, bn=True, **kwargs):
    super().__init__()
    self.conv1 = hk.Conv2D(channels, kernel_size, with_bias=not bn, w_init=init, **kwargs)
    self.conv2 = hk.Conv2D(channels, kernel_size, with_bias=not bn, w_init=init, **kwargs)
    if bn:
        self.bn1 = hk.BatchNorm(True, True, 0.999)
        self.bn2 = hk.BatchNorm(True, True, 0.999)
    else:
        self.bn1 = dummy_bn
        self.bn2 = dummy_bn

  def __call__(self, x, is_training):
    skip = x
    x = self.conv1(act(self.bn1(x, is_training)))
    x = self.conv2(act(self.bn2(x, is_training)))
    return skip + x


class TransposeBlock(hk.Module):
  def __init__(self, channels, kernel_shape, stride):
    super().__init__()
    self.bn = hk.BatchNorm(True, True, 0.999)
    self.conv = hk.Conv2DTranspose(channels, kernel_shape, stride, with_bias=False, w_init=init)

  def __call__(self, x, is_training):
    x = self.conv(x)
    x = self.bn(x, is_training=is_training)
    x = act(x)
    return x


class DownBlock(hk.Module):
  def __init__(self, channels, kernel_shape, stride):
    super().__init__()
    self.conv = hk.Conv2D(channels, kernel_shape, stride, w_init=init)
    self.channels = channels

  def __call__(self, x, is_training):
    x = self.conv(x)
    x = act(x)
    return x


class LinearBNRelu(hk.Module):
  def __init__(self, channels):
    super().__init__()
    self.bn = hk.BatchNorm(True, True, 0.999)
    self.fc = hk.Linear(channels, with_bias=False, w_init=init)

  def __call__(self, x, is_training):
    x = self.fc(x)
    x = self.bn(x, is_training=is_training)
    x = act(x)
    return x


class UpBlock(hk.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = hk.Conv2D(channels, 3, with_bias=False)
        self.conv2 = hk.Conv2D(channels, 3, with_bias=False)
        self.skip_conv = hk.Conv2D(channels, 1, with_bias=False)

        self.bn1 = hk.BatchNorm(True, True, 0.999)
        self.bn2 = hk.BatchNorm(True, True, 0.999)

    def __call__(self, x, is_training):
        """
        Largely inspired by https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
        """
        B, H, W, C = x.shape
        x = jax.image.resize(x, [B, H*2, W*2, C], 'linear')

        skip = x

        x = self.conv1(act(self.bn1(x, is_training)))
        x = self.conv2(act(self.bn2(x, is_training)))

        x = x + self.skip_conv(skip)

        return x


class DownBlock(hk.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = hk.Conv2D(channels, 3, with_bias=False)
        self.conv2 = hk.Conv2D(channels, 3, with_bias=False)
        self.skip_conv = hk.Conv2D(channels, 1, with_bias=False)

    def __call__(self, x):
        """
        Largely inspired by https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
        """

        x = hk.avg_pool(x, 2, 2, 'VALID')

        skip = x

        x = self.conv1(act(x))
        x = self.conv2(act(x))

        x = x + self.skip_conv(x)

        return x


class Generator():
    def __init__(self, output_channels):
        self.output_channels = output_channels

    def init(self):
        super().__init__()
        self.init_latent = hk.Linear(4 * 4 * 256)
        self.init_cond = hk.Conv2D(8, 3)

        self.blocks = [
            UpBlock( 256),
            UpBlock( 128),
            UpBlock( 128),
            UpBlock(  64),
            UpBlock(  32),
        ]

        self.cond_blocks = [
            DownBlock( 8),
            DownBlock(16),
            DownBlock(32),
            DownBlock(64),
            DownBlock(64),
        ]

        self.final_bn = hk.BatchNorm(True, True, 0.999)
        self.final_layer = hk.Conv2D(self.output_channels, 3)

    def __call__(self, x, cond, is_training):
        self.init()

        x = self.init_latent(x)
        x = rearrange(x, 'B (H W C) -> B H W C', H=4, W=4, C=256)

        cond = self.init_cond(cond)
        cond_connections = []
        for i, block in enumerate(self.cond_blocks):
            cond = block(cond)
            cond_connections.append(cond)

        for block, cond in zip(self.blocks, reversed(cond_connections)):
            x = jnp.concatenate([x, cond], axis=-1)
            x = block(x, is_training)

        x = self.final_layer(act(self.final_bn(x, is_training)))
        return jax.nn.sigmoid(x)


class Discriminator():
    def init(self):
        self.init_conv = hk.Conv2D(16, 3)

        self.blocks = [
            DownBlock( 32),
            DownBlock( 64),
            DownBlock(128),
            DownBlock(128),
            DownBlock(256),
        ]

        self.final_fc = hk.Linear(1, with_bias=False)

    def __call__(self, x, is_training):
        self.init()

        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        x = jnp.sum(act(x), axis=[1,2])
        return self.final_fc(x)

