---
classes: wide
title: My first C++ open source contribution -- Stan Library
tags: c++ ml linear-algebra
header:
    teaser: "assets/posts/stan/math.png"
excerpt_separator: <!--more-->
---
Writing L1 and L2 vector norms with reverse- and forward-mode autodiff.
<!--more-->

I successfully got my first [C++ Pull Request](https://github.com/stan-dev/math/pull/2636) with ~430 lines of code merged into the [Stan Library](https://github.com/stan-dev), a popular statistical library used for Bayesian modeling and inference.

I spent a good chunk of Xmas break reading textbooks and watching tutorials on C++.
But, you can only learn so much by doing textbook exercises and watching YouTube videos, so I wanted to build something _real_ to solidify my understanding.
C++ seems to be the lingua franca of video game programming, and there are plenty of tutorials online about how to build simple games, but I'm not super into that.
It made more sense to me to work on a project where I could leverage my existing skills and domain expertise (numerical linear algebra, machine learning, probabilistic models).

This is what motivated me to write a tiny linear algebra matrix class to [build and train a neural network from scratch]({% post_url 2022-01-06-mlp-train-cpp %}).
While this was fun, the project felt more like a one-off rather than a "real" software project.
I watched an old [CppCon talk by Titus Winters](https://youtu.be/NOCElcMcFik), the C++ tech lead at Google, who described how writing software with long-term stability and maintainability requires a completely different mindset than what most start-ups are junior/student devs (i.e. me) are used to.
This inspired me to contribute to a longer-term, large open-source collaboration in order to learn more about things like complex C++ library builds and unit testing.

## Choosing and open source project

I was torn between trying to contribute to either the PyTorch, or Stan libraries.
There were pros and cons to each that I had to weigh.
I use PyTorch extensively in my day-to-day, so I'm quite familiar with how it works; however, its repo is a behemoth with thousands of contributors and outstanding pull requests, so it looks very easy to get lost in all the noise.
With Stan, on the other hand, I have far less experience using the library; but, I am reasonably familiar with the problem domain (probabilistic models and probabilistic programming), and the smaller community seemed less daunting to join.
I think what ultimately pushed me over the fence was Bob Carpenter, one of the main Stan developers who happens to work down the hall from me; he is very friendly + knowledgeable, and encouraged me to contribute to the library when I mentioned I wanted to hone my C++ skills.

## My contributions to the Stan Library

While most people interface with the Stan library using a higher-level scripting language (e.g. RStan, PyStan, etc.), the Stan framework itself is largely built on a C++ foundation.
At the heart of Stan is its Math library, which basically wraps the existing `Eigen` C++ linear algebra library and extends it with automatic differentiation capabilities.

There was an [outstanding `stan-dev/math`issue](https://github.com/stan-dev/math/issues/2562) from this past summer describing the need for L1 and L2 norms in the library.
It was tagged as as a `good first issue`, meaning that it was an ideal problem for newcomers to work on.

### The code I wrote

My contributions mainly revolved around different function templates for these L1 and L2 norms.
These had to be exhaustive for all possible use-cases in the library.

1. Templated L1 and L2 norm functions operating on Containers with underlying `std::is_arithmetic` types
2. L1 and L2 norm functions with reverse-mode autodiff capabilities
3. L2 and L2 norm functions with forward-mode autodiff capabilities
4. Extensive Google Test unit testing for all these new functions with standard-use and edge cases

## Things I learned

- The Stan Library team is super friendly and welcoming
  - The back-and-forth process from initial pull request to getting reviewed and merged into the main `develop` branch only took 4 days, Dec 30 - Jan 2, (i.e. they were kind enough to review over the New Year weekend).
- Google Test C++ unit testing framework
  - This is a very popular tool in C++ projects so I'm glad I'm getting more familiar with it.
  - I'm very familiar with `PyTest` for Python, and Google Test feels pretty similar so far.
- C++ Template metaprogramming tricks
  - `stan-dev/math` is a (mostly) header-only library so at times it felt like template metaprogramming olympics to get stuff to run.
  - The Stan devs were very helpful during the review process and guided me through the confusing template expressions and type-trait landscape.
- How to implement reverse and forward-mode autodifferentiation
  - It's funny, I feel like I've used and read about autodiff for years now but never actually had to sit down and implement it myself.
  - The derivatives of L1 and L2 norms that I implemented were pretty straightforward, but [matrixcalculus.org](http://www.matrixcalculus.org/) was a good resource to double-check my math.
  - Most contemporary machine learning uses reverse-mode autodiff (for backpropagation), so [this blog post](https://kenndanielso.github.io/mlrefined/blog_posts/3_Automatic_differentiation/3_4_AD_forward_mode.html) was a good resource to familiarize myself with implementing forward-mode.

My experience as a first-time contributor to Stan (and a first-time C++ contributor to anything at all) was great, and I'm looking forward to making more contributions in the future.
