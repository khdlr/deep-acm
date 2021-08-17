import jax.numpy as jnp
from einops import rearrange

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
