<!-- Project title and teaser image -->
<br />
<p align="center">

  <h1 align="center"><a href="https://rgl.epfl.ch/publications/NimierDavid2022Unbiased/">Unbiased Inverse Volume Rendering with Differential Trackers</a></h1>

  <a href="https://rgl.epfl.ch/publications/NimierDavid2022Unbiased/">
    <!-- TODO: dark mode version -->
    <img src="https://rgl.s3.eu-central-1.amazonaws.com/media/images/papers/NimierDavid2022Unbiased_teaser.jpg" alt="Logo" width="100%">
  </a>

  <p align="center">
    ACM Transactions on Graphics - 2022
    <br />
    <a href="https://merlin.ninja"><strong>Merlin Nimier-David</strong></a>
    ·
    <a href="https://tom94.net/"><strong>Thomas Müller</strong></a>
    ·
    <a href="https://research.nvidia.com/person/alex-keller"><strong>Alexander Keller</strong></a>
    ·
    <a href="https://rgl.epfl.ch/people/wjakob/"><strong>Wenzel Jakob</strong></a>
  </p>

  <p align="center">
    <a href='https://rgl.s3.eu-central-1.amazonaws.com/media/papers/NimierDavid2022Unbiased_3.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='https://rgl.epfl.ch/publications/NimierDavid2022Unbiased/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p>
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 15px; border-radius:5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a>
    <li><a href="#citation">Citation</a>
    <li><a href="#getting-started">Getting started</a>
    <li><a href="#running-an-optimization">Running an optimization</a>
    <li><a href="#limitations">Limitations</a>
    <li><a href="#implementation-details">Implementation details</a>
    <li><a href="#acknowledgements">Acknowledgements</a>
  </ol>
</details>

<br>

Overview
--------

This repository contains code examples to reproduce the results from the article:

> Merlin Nimier-David, Thomas Müller, Alexander Keller, and Wenzel Jakob. 2022.
> Unbiased Inverse Volume Rendering with Differential Trackers.
> In Transactions on Graphics (Proceedings of SIGGRAPH) 41(4).

It uses the [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) differentiable renderer.

Citation
--------

This code is released under the [BSD 3-Clause License](LICENSE). Additionally, if you are using this code in academic research, please cite our paper using the following BibTeX entry:

```bibtex
@article{nimierdavid2022unbiased,
    author = {Merlin Nimier-David and Thomas M\"uller and Alexander Keller and Wenzel Jakob},
    title = {Unbiased Inverse Volume Rendering with Differential Trackers},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {44:1--44:20},
    articleno = {44},
    numpages = {20},
    url = {https://doi.org/10.1145/3528223.3530073},
    doi = {10.1145/3528223.3530073},
    publisher = {ACM},
    address = {New York, NY, USA},
    keywords = {differentiable rendering, inverse rendering, volumetric rendering, radiative backpropagation, importance sampling}
}
```


Getting started
---------------

This code was tested on Ubuntu 20.04 with an NVIDIA Titan RTX GPU.
NVIDIA driver version 515.48.07 was used with CUDA 11.2.

Mitsuba 3 was compiled with Clang++ 12.0.0 and the provided scripts were run with Python 3.8.10.
The `cuda_ad_rgb` Mitsuba variant was selected, although the `llvm_ad_rgb` variant is also compatible in principle.

This implementation relies on modifications to the Mitsuba source code, which are available on the `unbiased-inverse-volume-rendering` branch of the `mitsuba3` repository.
**Please make sure to checkout the correct branch** as follows.
Note the `--recursive` and `--branch` flags:

```bash
# Cloning Mitsuba 3 and this repository
git clone --recursive https://github.com/mitsuba-renderer/mitsuba3 --branch unbiased-inverse-volume-rendering
git clone --recursive https://github.com/rgl-epfl/unbiased-inverse-volume-rendering

# Building Mitsuba 3, including the project-specific modifications
cd mitsuba3
mkdir build && cd build
cmake -GNinja ..
ninja
```

The `cuda_ad_rgb` and `llvm_ad_rgb` variants should be included by default.
Please see the [Mitsuba 3 documentation](https://mitsuba.readthedocs.io/en/latest/#) for complete instructions on building and using the system.

The scene data must be downloaded and unzipped at the root of the project folder:

```bash
cd unbiased-inverse-volume-rendering
wget https://rgl.s3.eu-central-1.amazonaws.com/media/papers/NimierDavid2022Unbiased.zip
unzip NimierDavid2022Unbiased.zip
rm NimierDavid2022Unbiased.zip
ls scenes
# The available scenes should now be listed (one directory per scene)
```


Running an optimization
-----------------------

Navigate to this project's directory and make sure that the Mitsuba 3 libraries built in the previous step are made available in your current session using `setpath.sh`:

```bash
cd unbiased-inverse-volume-rendering
source ../mitsuba3/build/setpath.sh
# The following should execute without error and without output
# (use the variant 'llvm_ad_rgb' if your system does not support the CUDA backend):
python3 -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb')"
```

From here, the script `python/reproduce.py` can be used to run inverse volume rendering examples using different methods.

Each **method** is exposed as an `IntegratorConfig` in `python/opt_config.py`:

- `nerf`: uses a non-physical emissive volume model. It is highly simplified compared to true NeRFs, with no support for directionally-varying emission and is backed by a regular grid rather than a neural network.
- `volpathsimple-basic`: corresponds to the baseline physically based method. It uses path replay backropagation and the standard free-flight path sampling technique to estimate gradients. It is susceptible to bias and high variance in regions where the medium density approaches zero.
- `volpathsimple-drt`: our _differential ratio tracking_ method. It samples in-scattering gradients with a dedicated estimator. Multiple importance sampling is used to combine the reesults with the baseline.

Next, **scene configurations** are defined in `python/scene_config.py`.
They include the scenes showcased in the papers and specify various parameters such as the rendering resolution, the initialization value, which sensors (camera viewpoints) to include in the optimization, and the name of parameters to optimize.

Finally, **optimization configurations** include all other parameters such as the iteration and sample counts, the loss function, whether to save parameter values at regular intervals, etc.
They can be freely defined by the user, but appropriate values of these parameters are provided in `python/reproduce.py` for each scene.

The script can be invoked without arguments, in which case it will render all reference images and run all existing inverse rendering experiments:

```bash
python3 ./python/reproduce.py
```

Alternatively, a specific combination of scene and method can be specified.
In this case, the optimization parameters will be looked up from the `reproduce.py` file.

```bash
python3 ./python/reproduce.py --config rover-sn64 --integrator volpathsimple-drt
```


Limitations
-----------

For ease of implementation and increased performance, we have implemented simplified volumetric path tracers that make the following important assumptions:

- There are no surfaces in the scene!
- There is only one medium in the scene, contained within a convex bounding volume.
- The medium boundary must use a `null` BSDF
- The only emitter is an infinite light source (e.g. `envmap` or `constant`).


Implementation details
----------------------

Our Differential Ratio Tracking algorithm, as well as the baseline free-flight sampling-based method are implemented in the [`VolpathSimpleIntegrator` class](blob/master/python/integrators/volpathsimple.py), which implements a Mitsuba 3 integrator plugin.

As such, it can be loaded and used as any other Mitsuba integrator:

```python
import sys
sys.path.append('./python')
# The import must take place for the plugin to become visible to Mitsuba
from integrators.volpathsimple import VolpathSimpleIntegrator


import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

integrator = mi.load_dict({
    'type': 'volpathsimple',
    'max_depth': 64,
    'use_drt': True,
})

image = mi.render(scene, integrator=integrator, ...)
```

The integrator supports the following main parameters:

| Parameter name        | Type   | Usage       |
|-----------------------|--------|-------------|
| `use_nee`             | `bool` | Enables Next-Event Estimation in both the primal and adjoint rendering phases
| `use_drt`             | `bool` | Enables our method, Differential Ratio Tracking. Otherwise, uses the free-flight sampling-based baseline.
| `use_drt_subsampling` | `bool` | Reduces usage of our sampling method to once per path rather than once at every path vertex. This reduces the overall runtime from `O(n^2)` to `O(n)`.
| `use_drt_mis`         | `bool` | Combines the gradients estimated with our method and the baseline using MIS.

Important support code is also provided to reproduce the inverse rendering optimizations:

- `python/optimize.py`: main optimization loop, reference image rendering, checkpointing, parameter upsampling (multiresolution), etc.
- `python/batched.py`: provides the `render_batch` function as an alternative to `mi.render`. Instead of rendering a single sensor at a time, samples a batch of rays from a collection of sensors and renders them all at once.


Acknowledgements
----------------

The experiment handling framework and repository structure was inspired by Delio Vicini's code.
The format of this README was adapted from [Miguel Crespo](https://github.com/mcrescas/viltrum-mitsuba/blob/457a7ffbbc8b8b5ba9c40d6017b5d08f0f41a886/README.md).

Volumes, environment maps and 3D models were generously provided by [JangaFX](https://jangafx.com/software/embergen/download/free-vdb-animations/), [PolyHaven](https://polyhaven.com/hdris), Antoan Shiyachki, [jgilhutton](https://blendswap.com/blend/12622), [vajrablue](https://blendswap.com/blend/28458) and [Zuendholz](https://blendswap.com/blend/3319).
