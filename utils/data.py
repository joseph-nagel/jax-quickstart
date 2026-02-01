'''Data creation.'''

import jax
import jax.numpy as jnp


def make_linear_data(
    key,
    num_samples,
    W,
    b,
    noise_level=None
):
    '''Create data according to linear model.'''

    x_dim, y_dim = W.shape

    key_x, key_noise = jax.random.split(key)

    # sample inputs
    x = jax.random.normal(key_x, (num_samples, x_dim))

    # compute targets
    y_perfect = jnp.dot(x, W) + b

    # add noise
    if noise_level is not None:
        eps = jax.random.normal(key_noise, (num_samples, y_dim))
        y = y_perfect + abs(noise_level) * eps
    else:
        y = y_perfect

    return x, y
