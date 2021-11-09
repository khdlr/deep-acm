import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.experimental import host_callback
from skimage.measure import find_contours
from functools import partial

def distance_matrix(a, b):
    a = rearrange(a, '(true pred) d -> true pred d', true=1)
    b = rearrange(b, '(true pred) d -> true pred d', pred=1)
    D = jnp.sum(jnp.square(a - b), axis=-1)
    return D


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def fmt_num(x):
    if jnp.isinf(x):
        return '∞'.rjust(8)
    else:
        return f'{x:.2f}'.rjust(8)


def fmt(xs, extra=None):
    tag = ''
    if isinstance(xs, str):
        tag = xs
        xs = extra
    rank = len(xs.shape)
    if rank == 1:
        print(tag, ','.join([fmt_num(x) for x in xs]))
    elif rank == 2:
        print('\n'.join(','.join([fmt_num(x) for x in row]) for row in xs))
        print()


def snakify(mask, vertices):
    res = host_callback.call(snakify_host, (mask, vertices),
            result_shape=jnp.zeros([mask.shape[0], vertices, 2], jnp.float32))
    return res


def snakify_host(args):
    masks, vertices = args
    res = np.zeros([masks.shape[0], vertices, 2], np.float32)
    for i, mask in enumerate(masks):
        contours = find_contours(mask[..., 0], 0)
        # Select the longest contour
        if len(contours) == 0:
            continue

        contour = max(contours, key=lambda x: x.shape[0])
        contour = contour.astype(np.float32)
        contour = contour.view(np.complex64)[:, 0]
        C_space = np.linspace(0, 1, len(contour), dtype=np.float32)
        S_space = np.linspace(0, 1, vertices, dtype=np.float32)
        snake = np.interp(S_space, C_space, contour)
        snake = snake[:, np.newaxis].view(np.float64).astype(np.float32)

        res[i] = snake * (2.0 / mask.shape[0]) - 1.0
    return res
