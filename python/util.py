"""
Various helper functions.
"""
import os
import pickle

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


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result
