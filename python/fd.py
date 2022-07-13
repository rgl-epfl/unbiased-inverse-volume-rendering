from os.path import join

import drjit as dr
import mitsuba as mi
import numpy as np
from tqdm import tqdm


def fd_gradients(output_dir, scene, params, loss_fn, eps,
                 spp=4096, write_images=False, integrator=None, seed = 1234):
    param_names = list(params.keys())
    im_center = mi.render(scene, spp=spp, seed=seed, integrator=integrator)
    loss_center = loss_fn(im_center)
    if write_images:
        fname = join(output_dir, f'fd_center.exr')
        mi.Bitmap(im_center) \
            .convert(component_format=mi.Struct.Type.Float32) \
            .write(fname)
        print(f'[+] {fname}')

    results = {}
    for run_i, k in enumerate(tqdm(param_names, desc='Finite differences')):
        v_original = params[k].numpy()
        grads = np.full(v_original.shape, np.nan)

        if v_original.size == 3:
            suffixes = 'rgb'
            indices = [(i,) for i in range(len(suffixes))]
        else:
            indices = [np.unravel_index(i, v_original.shape) for i in range(v_original.size)]
            suffixes = ['_'.join([str(ii) for ii in idx]) for idx in indices]

        for ci, k_suffix in tqdm(list(zip(indices, suffixes)), desc='Param entries'):
            v = v_original.copy()
            if len(ci) == 1:
                v[ci] = v_original[ci] + eps
            elif len(ci) == 3:
                v[ci[0], ci[1], ci[2], ...] = v_original[ci[0], ci[1], ci[2], ...] + eps
            elif len(ci) == 4:
                v[ci[0], ci[1], ci[2], ci[3]] = v_original[ci[0], ci[1], ci[2], ci[3]] + eps
            else:
                raise NotImplementedError()
            params[k] = type(params[k])(v)
            params.update()
            im_offset = mi.render(scene, spp=spp, seed=seed, integrator=integrator)
            loss_offset = loss_fn(im_offset)
            g = (loss_offset - loss_center) / (eps)

            # Output gradient value
            if len(ci) == 1:
                grads[ci] = g.numpy().item()
            elif len(ci) == 3:
                grads[ci[0], ci[1], ci[2], ...] = g.numpy().item()
            elif len(ci) == 4:
                grads[ci[0], ci[1], ci[2], ci[3]] = g.numpy().item()

            if write_images:
                fname = join(output_dir, f'fd_{run_i}_{k_suffix}.exr')
                mi.Bitmap(im_offset) \
                  .convert(component_format=mi.Struct.Type.Float32) \
                  .write(fname)
                print(k, ci, k_suffix, fname)

        results[k] = grads
        # Restore parameter before moving on
        params[k] = type(params[k])(v_original)
        params.update()

    return results
