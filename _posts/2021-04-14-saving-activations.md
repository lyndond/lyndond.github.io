---
classes: wide
title: Saving PyTorch activations w/ forward hooks
image: /assets/images/generic_code.png
tags: ml python pytorch
header:
    teaser: "/assets/images/generic_code.png"
excerpt_separator: <!--more-->
---
Simple code to save activations of a model's intermediate layers.
<!--more-->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyndond/lyndond.github.io/blob/master/code/2021-04-14-saving-activations.ipynb)
[![Open on GitHub](https://img.shields.io/badge/Open on GitHub-success.svg)](https://github.com/lyndond/lyndond.github.io/blob/master/code/2021-04-14-saving-activations.ipynb)

I recently had to run a trained model and save intermediate-layer activations.
This is some simple code to do that using torch forward hooks.

First define the hook:

```python
import torch
import torch.nn.functional as F
from torch import nn
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial

def save_activations(
        activations: DefaultDict,
        name: str,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    activations[name].append(out.detach().cpu())
```

Then define a helper method that registers the hook to specified layers of a model
(requires `functools.partial`).

```python
def register_activation_hooks(
        model: nn.Module,
        layers_to_save: List[str]
) -> DefaultDict[List, torch.Tensor]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    layers_to_save:
        Module names within ``model`` whose activations we want to save.

    Returns
    -------
    activations_dict:
        dict of lists containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = collections.defaultdict(list)

    for name, module in model.named_modules():
        if name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations, activations_dict, name)
            )
    return activations_dict
```

Now define a simple model to test out `register_activation_hooks()`:

```python
class Net(nn.Module):
    """Simple two layer conv net"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(2,2))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(2,2))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        z = F.relu(self.conv2(y))
        return z

mdl = Net()
to_save = ["conv1", "conv2"]

# register fwd hooks in specified layers
saved_activations = register_activation_hooks(mdl, layers_to_save=to_save)

# run twice, then assert each created lists for conv1 and conv2, each with length 2
num_fwd = 2
images = [torch.randn(10, 3, 256, 256) for _ in range(num_fwd)]
for _ in range(num_fwd):
    mdl(images[_])

assert len(saved_activations["conv1"]) == num_fwd
assert len(saved_activations["conv2"]) == num_fwd
```
