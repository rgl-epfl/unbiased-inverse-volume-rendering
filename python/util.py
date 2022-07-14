"""
Various helper functions.
"""
import os
import pickle

import mitsuba as mi

def pickle_cache(fname, overwrite=False):
    """Cache results of long-running functions."""
    def decorator(fn):
        def decorated(*args, **kwargs):
            if (not overwrite) and os.path.exists(fname):
                with open(fname, 'rb') as f:
                    return pickle.load(f)
            else:
                result = fn(*args, **kwargs)
                with open(fname, 'wb') as f:
                    pickle.dump(result, f)
                return result
        return decorated

    return decorator

def render_cache(fname, overwrite=False, verbose=True):
    """Cache results of long-running rendering functions."""
    def decorator(fn):
        def decorated(*args, **kwargs):
            if (not overwrite) and os.path.exists(fname):
                if verbose:
                    print(f'[↑] {fname}')
                return mi.Bitmap(fname)
            else:
                result = fn(*args, **kwargs)
                mi.Bitmap(result).write(fname)
                if verbose:
                    print(f'[+] {fname}')
                return result
        return decorated

    return decorator


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def save_params(output_dir, scene_config, params, name):
    for key in scene_config.param_keys:
        value = params[key]
        if not key.endswith('.data'):
            # TODO: support saving scalar parameters
            raise NotImplementedError(f'Checkpointing of parameter {key} with type {type(value)}')

        # Heuristic to get the variable name from a parameter key.
        for suffix in ['.data', '.values', '.value']:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
        var_name = '_'.join(key.strip().split('.'))

        fname = os.path.join(output_dir, f'{name}-{var_name}.vol')
        # TODO: check this doesn't mix-up the dimensions (data ordering)
        grid = mi.VolumeGrid(value.numpy())
        grid.write(fname)
