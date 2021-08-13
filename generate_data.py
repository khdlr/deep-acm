import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Tuple


def get_intersection(start_point, end_point, x):
    d = end_point - start_point
    slope = d[0] / d[1]
    y = start_point[0] + (x - start_point[1]) * slope
    # Emulating an xnor here
    isinside = (start_point[1] - x >= 0) == (x - end_point[1] > 0)
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
    centerpoints = (start_points + end_points) / 2 + 0.2 * length * \
            jax.random.truncated_normal(key, -2., 2., start_points.shape)
    all_points_but_last = jnp.stack([start_points, centerpoints], axis=1).reshape(-1, 2)
    all_points = jnp.concatenate([all_points_but_last, end_points[-1:]], axis=0)
    return all_points


def generate_image(key: jnp.array) -> Tuple[jnp.array, jnp.array]:
    key, k_theta1, k_theta2, k_center = jax.random.split(key, 4)
    theta1 = jax.random.uniform(k_theta1, (), minval=0.0, maxval=2*jnp.pi)
    theta2 = jax.random.uniform(k_theta2, (), minval=theta1+np.deg2rad(30), maxval=theta1+np.deg2rad(360-30))
    theta = jnp.stack([theta1, theta2])
    centerpoint = jax.random.uniform(k_center, (1, 2), minval=-1.0, maxval=1.0)

    startend = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    p0, pn = jnp.split(startend / jnp.max(jnp.abs(startend), axis=1, keepdims=True), 2, axis=0)
    polyline = jnp.concatenate([p0, centerpoint, pn], axis=0)

    for i in range(5):
        key, subkey = jax.random.split(key)
        polyline = subdivide_and_noise(polyline, subkey)

    coords = jnp.linspace(-1, 1, 128)

    # Build an outer ring to ensure the polygon being correctly filled
    lerp = jnp.linspace(0.0, 1.0, 4)
    lerp = jnp.stack([lerp, 1-lerp], axis=1)
    # theta = theta + jnp.where(theta[1] < theta[0], jnp.array([0, 2*jnp.pi]), jnp.zeros(2))
    theta = theta.reshape(1, -1)
    outer_theta = (lerp * theta).sum(axis=1)
    outer_points = 5 * jnp.stack([jnp.cos(outer_theta), jnp.sin(outer_theta)], axis=1)

    polyring = jnp.concatenate([
        polyline, 
        outer_points,
        polyline[:1]
    ], axis=0)
    image = fill_scanline(polyring, coords, coords)
    image = jnp.expand_dims(image, -1).astype(jnp.float32)

    polyline = jnp.maximum(-1., jnp.minimum(1., polyline))

    return image, polyline
