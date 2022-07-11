---
classes: wide
title: Reproducible latents
tags: ml pytorch
header:
    teaser: "assets/posts/reproducible_latents/mnist_2d.png"
excerpt_separator: <!--more-->
---

Can we rely on VAEs to generate reproducible latents?

<!--more-->

[Variational autoencoders](https://arxiv.org/abs/1312.6114) (VAEs) originated as a  method for learning probabilistic generative models of data.
In recent years, there have been countless studies using VAEs as tools to infer low-dimensional latent structure from high-dimensional data.
The nonlinearity/flexibility of these models begs the question: how reliable are VAEs in uncovering these latents?
This is especially important if we want to use them to draw hard scientific conclusions from our data.

Below are different latents from the exact same network architecture (MLP with ReLU non-linearities), but with 5 different random initialization seeds to auto-encode MNIST digits using the standard VAE objective.
Each dot is a different example and different colors are different digits.

![latent](/assets/posts/reproducible_latents/mnist_2d.png)

All of the converged reconstruction errors were effectively the same, but the latents are wildly different (even after rotating/scaling to minimize squared error).

There are ways to remedy this, related to ideas of [identifiability](https://en.wikipedia.org/wiki/Identifiability), which has recently gained popularity in latent variable and generative modelling.
[(I even implemented an identifiable model for spiking neural data in a previous note.)]({% post_url 2021-11-25-pivae %})
For this note, I just wanted to show a simple, compelling example of how different the solutions can be.
As researchers and practicioners we should take care to re-run the same experiments with a bunch of different seeds to ensure we get the same qualitative trends in the latents.
