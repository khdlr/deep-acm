import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Tuple


def get_intersection(start_point, end_point, x):
    d = end_point - start_point
    slope = d[0] / d[1]
    y = start_point[0] + (x - start_point[1]) * slope
    isinside = (start_point[1] - x) * (x - end_point[1]) >= 0
    y = jnp.where(isinside, y, jnp.inf)
    return y


def scan_column(polyline, y_vals, x):
    start_points = polyline[:-1]
    end_points = polyline[1:]

    intersections = jax.vmap(get_intersection, [0, 0, None])(start_points, end_points, x)

    y_vals        = rearrange(y_vals,        '(edge y_val) -> edge y_val', edge=1)
    intersections = rearrange(intersections, '(edge y_val) -> edge y_val', y_val=1)

    should_fill = jnp.sum(y_vals >= intersections, axis=0) % 2 == 0
    return should_fill


fill_scanline = jax.vmap(scan_column, [None, None, 0], 1)


def subdivide_and_noise(line, key):
    start_points = line[:-1]
    end_points = line[1:]
    length = jnp.linalg.norm(start_points - end_points, axis=1, keepdims=True)
    centerpoints = (start_points + end_points) / 2 + 0.2 * length * jax.random.normal(key, start_points.shape)
    all_points_but_last = jnp.stack([start_points, centerpoints], axis=1).reshape(-1, 2)
    all_points = jnp.concatenate([all_points_but_last, end_points[-1:]], axis=0)
    return all_points


def generate_image(key: jnp.array) -> Tuple[jnp.array, jnp.array]:
    key, k_theta, k_center = jax.random.split(key, 3)
    theta = jax.random.uniform(k_theta, (2, ), minval=0.0, maxval=2*jnp.pi)
    centerpoint = jax.random.uniform(k_center, (1, 2), minval=-1.0, maxval=1.0)

    startend = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    p0, pn = jnp.split(startend / jnp.max(jnp.abs(startend), axis=1, keepdims=True), 2, axis=0)
    polyline = jnp.concatenate([p0, centerpoint, pn], axis=0)

    for i in range(5):
        key, subkey = jax.random.split(key)
        polyline = subdivide_and_noise(polyline, subkey)

    coords = jnp.linspace(-1, 1, 128)
    polyline_long = jnp.concatenate([
        polyline[:1] + (polyline[:1] - polyline[1:2]) * 1000,
        polyline[1:-1],
        polyline[-1:] + (polyline[-1:] - polyline[-2:-1]) * 1000,
    ], axis=0)
    image = fill_scanline(polyline_long, coords, coords)
    image = jnp.expand_dims(image, -1).astype(jnp.float32)

    return image, polyline
