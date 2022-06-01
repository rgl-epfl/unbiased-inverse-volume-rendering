Unbiased Inverse Volume Rendering with Differential Trackers
============================================================

This repository contains code examples to reproduce the results from the article:

> Merlin Nimier-David, Thomas Müller, Alexander Keller, and Wenzel Jakob. 2022.
> Unbiased Inverse Volume Rendering with Differential Trackers.
> In Transactions on Graphics (Proceedings of SIGGRAPH) 41(4).


Getting started
---------------

This code was tested on Ubuntu 20.04 with an NVIDIA Titan RTX GPU.
Mitsuba 3 was compiled with Clang++ 9.0.1 and scripts run with Python 3.8.10.
NVIDIA driver version 460.91 was used with CUDA 11.2.

```
git clone --recursive TODO/mitsuba3
git clone --recursive TODO/this-repo
cd mitsuba3
mkdir build && cd build
cmake -GNinja ..
# TODO: enable cuda_ad_rgb variant
ninja
```

Download the scenes.

Running an optimization
-----------------------

Configs are defined in the file: TODO

```
cd this-repo
source ../mitsuba3/build/setpath.sh
python3 ./python/run_optimization.py "janga-smoke-sn64-d64" --spp=16 --primal_spp_mult=64 --learning_rate=5e-3 --n_iter=6000 --configs volpathsimpledrt --batch_size=32768
```


Acknowledgements
----------------

TODO Volumes, meshes, textures, envmaps
