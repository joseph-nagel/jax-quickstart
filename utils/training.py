'''Model training.'''

import jax.numpy as jnp


def mse(preds, targets):
    '''Compute MSE loss.'''
    diffs = targets - preds
    loss = jnp.mean(diffs**2)
    return loss

