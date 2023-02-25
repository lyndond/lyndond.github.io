---
classes: wide
title: Python QR code generator for poster presentations
tags: python
header:
    teaser: "assets/posts/qr_code_python/qr_py.png"
excerpt_separator: <!--more-->
---
Code snippet for generating QR codes with transparent backgrounds.
<!--more-->

Code for this post:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyndond/lyndond.github.io/blob/master/code/2023-02-13-qr-code-python.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open on GitHub-success.svg)](https://github.com/lyndond/lyndond.github.io/blob/master/code/2023-02-13-qr-code-python.ipynb)

I wanted to put a QR code linking to an arXiv preprint on my CoSyNe poster but the online solutions all had ugly white backgrounds. So I wrote a Python snippet to generate a QR code with **transparent whitespace**. It creates a QR code, converts it to a numpy RGBA array with the alpha channel set according to whether or not there is a black pixel.

```python
!pip install qrcode

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import qrcode

def make_qr_png(
    url: str,
    filename: Optional[os.PathLike] = 'qr_py.png',
    dpi: Optional[int] = 300,
    ) -> None:
  """Saves a png QR code to a URL with transparent whites and background.
  Parameters
  ----------
  url: url that the QR code should point to.
  filename: For saving.
  dpi: matplotlib figure dpi.
  """

  qr = qrcode.QRCode()
  qr.add_data(url)
  img = qr.make_image()

  # cast to numpy
  img = np.array(img)
  h, w = img.shape
  rgb = [img for _ in range(3)]
  rgb = np.stack(rgb, -1)
  alpha = np.zeros(rgb.shape[:2])

  # set alpha channel 
  rgba = np.zeros((h, w, 4))
  rgba[...,:3] = rgb
  rgba[...,-1] = (1 - np.max(rgba[...,:3], axis=-1))*255
  rgba = rgba.astype(int)

  #  
  fig, ax = plt.subplots(1, 1, dpi=dpi)
  ax.imshow(rgba)
  ax.axis('off')
  fig.savefig(filename, transparent=True)

  
url = "doi.org/10.48550/arXiv.2301.11955"  # set to whatever; shorter urls make cleaner QR codes
make_qr_png(url)
```
