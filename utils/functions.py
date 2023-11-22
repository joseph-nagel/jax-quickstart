'''JAX functions.'''

import jax.numpy as jnp


def sigmoid(x):
    '''Compute sigmoid function.'''
    return 1 / (1 + jnp.exp(-x))


def sigmoid_sum(x):
    '''Compute sigmoid and sum outputs.'''
    sigma = sigmoid(x)
    return jnp.sum(sigma)


def sigmoid_deriv(x):
    '''Compute derivative of the sigmoid.'''
    sigma = sigmoid(x)
    deriv = sigma * (1 - sigma)
    return deriv

