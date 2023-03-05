---
classes: wide
title: Statistical whitening with 1D projections
tags: geometry probabillity
header:
    teaser: "assets/posts/1d-proj-gaussian/sliced.png"
excerpt_separator: <!--more-->
---
Geometric intuition for our recent paper on statistical whitening using overcomplete bases.
<!--more-->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyndond/lyndond.github.io/blob/master/code/2023-03-05-1d-proj-gaussian.ipynb)
[![Open on GitHub](https://img.shields.io/badge/Open on GitHub-success.svg)](https://github.com/lyndond/lyndond.github.io/blob/master/code/2023-03-05-1d-proj-gaussian.ipynb)

<!-- describe statistical whitening and history -->
A very old problem in statistics and signal processing is to **statistically whiten** a signal, i.e. to linearly transform a signal with covariance $${\bf C}$$ to one with identity covariance. The most common approach to do this to find the principal components of the signal (the eigenvectors of $${\bf C}$$), then scale the signal according to how much the signal varies along each principal axis. The downside of this approach is that if the inputs _change_, then the principal axes need to be recomputed.

In our recent paper [(https://arxiv.org/abs/2301.11955)](https://arxiv.org/abs/2301.11955), we introduce a **completely different and novel approach** to statistical whitening. We do away with finding principal components altogether, and instead develop a framework for whitening with a **fixed, overcomplete basis** using concepts borrowed from [frame theory of linear algebra](https://en.wikipedia.org/wiki/Frame_(linear_algebra)), and [tomography](https://en.wikipedia.org/wiki/Tomography), the science of reconstructing signals from projections.

![png](/assets/posts/1d-proj-gaussian/1d_intuition.png)

The figure above shows the geometric intuition behind our approach. It's useful to know that we can geometrically represent densities with covariance matrices $${\bf C}$$ as ellipsoids in $$N$$-dimensional space (top left panel, shaded black). [Old work in tomography](https://doi.org/10.1006/cgip.1994.1012) has shown that ellipsoids can be represented (reconstructed) from a series of 1D projections. The 1D projected densities are plotted as vertical lines in the middle panel, with colors corresponding to the axes along which the original density was projected, and color saturation denoting probability at a given point. It turns out that if the density is Gaussian, then $$N(N+1)/2$$ projections along unique axes are **necessary and sufficient** to represent the original density; the number of required projections is the the number of independent parameters in a covariance matrix. Notably, the set of 1D projection axes _can exclude the principal components_ of the original density, and is **overcomplete**, i.e. linearly dependent, since there are more than $$N$$ projections.

Unlike conventional tomographic approaches, the main goal of our study isn't to reconstruct the ellipsoid, but rather to use the information derived from its projections to **whiten** the original signal. The top right plot shows each 1D density's variance; notice how the variance of the 1D projections is proportional to the length of the corresponding 1D slice, and that for this non-white density, the variances are quite variable. Meanwhile, for a whitened signal (bottom row), **all projected variances are all equal to 1**! This geometric intuition involving 1D projections of Gaussian densities forms the foundation of our framework.

In the [paper](https://arxiv.org/abs/2301.11955) we show: 1) how to operationalize these geometric ideas into an optimization objective function for statistical whitening; and 2) how to derive a recurrent neural network that iteratively optimizes this objective and converges to a steady-state solution where the outputs of the network are statistically white. From a theoretical neuroscience perspective, our findings are particularly exciting because the generalize well-established ideas of single-neuron adaptive efficient coding via gain control to the level of the entire neural population.
