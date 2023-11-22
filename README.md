# JAX quickstart

Let's get started with [JAX](http://jax.readthedocs.io/), a high-performance computing framework.
It has a NumPy-like API, is GPU/TPU-capable and features built-in automatic differentiation.
JAX might not be as mature as other frameworks, say PyTorch with its great community and ecosystem,
but it has recently gained momentum due to its performance in some scientific applications.

Essentially, JAX is a framework for composing functions and function transformations.
Autograd and JIT-compilation may be the two most important examples.
Programs need to be written according to a certain functional programming paradigm,
in order to fully benefit from JAX's computational machinery.

Even though JAX is not a deep learning tool per se,
it is certainly well-suited for typical tasks involving neural networks.
One can therefore build dedicated higher-level libraries on top of JAX.
An example of such a JAX-based library is [Flax](https://flax.readthedocs.io).

## Instructions

The CPU-only version of JAX can be installed via `pip install jax[cpu]`.

