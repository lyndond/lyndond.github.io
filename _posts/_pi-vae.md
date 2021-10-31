---
title: Poisson Identifiable Variational Auto-Encoder
classes: wide
tags: probability ml pytorch vae
header:
    teaser: ""
excerpt_separator: <!--more-->
---

Learning identifiable latent structure in data, implemented in PyTorch.
<!--more-->

[PyTorch code available here](https://github.com/lyndond/lyndond.github.io/blob/master/code/2021-02-16-orthogonal-iteration.ipynb){: .btn .btn--success .btn--large}

I saw a talk on a recently-published [paper](https://arxiv.org/abs/2011.04798) applying a form of Variational Auto-Encoder (VAE) on neural data to extract latent structure.
VAEs provide an efficient way to learn deep latent variable models that accurately describe observed data.
Often we want to not just characterize the observed data but also learn the joint distribution over the observed and latent variables -- including the true posterior and prior over latents.
This is a non-trivial task and is in fact [not possible without adding complexity to the model](https://arxiv.org/abs/1907.04809).

