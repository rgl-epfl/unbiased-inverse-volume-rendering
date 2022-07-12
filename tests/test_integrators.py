import os
from os.path import realpath, join, dirname
import sys

extras = [realpath(join(dirname(__file__), '../python'))]
for p in extras:
    if p not in sys.path:
        sys.path.insert(0, p)

import drjit as dr
import mitsuba as mi
import numpy as np

from constants import SCENE_DIR, OUTPUT_DIR
from fd import fd_gradients
from util import pickle_cache, render_cache, gallery


def cube_test_scene(resx=128, resy=128, spp=16, pixel_format='rgb', sample_emitters=True,
                    density_scale=1.0, use_fast_path=False):
    T = mi.ScalarTransform4f

    grids = [np.full((3, 3, 3, k), 1.0, dtype=np.float32) for k in (1, 3)]
    # Add some basic spatial variations
    grids[0] *= 0.5
    grids[0][0, 0, 0, :] = 0.1
    grids[0][0, -1, 0, :] = 2.0
    grids[0][0, 0, -1, :] = 0.2
    grids[1][..., 0] = 0.3
    grids[1][..., 1] = 0.5
    grids[1][..., 2] = 0.9
    for i in range(grids[1].shape[0]):
        grids[1][i, :, :, 0] *= np.square((i+1) / grids[1].shape[0])
        grids[1][i, :, :, 1] *= 1 - (i+1) / grids[1].shape[0]
        grids[1][:, i, :, 1] *= np.square((i+1) / grids[1].shape[0])
    # Albedo grid
    grids.append(np.clip(grids[1], 0, 1))

    grids = [mi.VolumeGrid(g) for g in grids]
    to_world = T.translate([-0.5, -0.5, -0.5]).scale([2, 2, 2])

    return {
        'type': 'scene',
        'use_bbox_fast_path': use_fast_path,
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov': 30,
            'to_world': T.look_at(
                origin=[4.0, 4.0, 4.0],
                target=[0, -0.15, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
            'film': {
                'type': 'hdrfilm',
                'width' : resx,
                'height': resy,
                'rfilter': {
                    'type': 'box',
                },
                'pixel_format': pixel_format,
            }
        },
        # Mostly just to avoid the warning
        'integrator': {
            'type': 'path',
        },
        # -------------------- Light --------------------
        'light': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1.0, 0.8, 0.2]},
            # TODO: switch to an envmap
        },
        # -------------------- Media --------------------
        'medium1': {
            'type': 'heterogeneous',
            'sample_emitters': sample_emitters,
            'has_spectral_extinction': False,
            'scale': density_scale,
            # 'albedo': {
            #     'type': 'constvolume',
            #     'value': {'type': 'rgb', 'value': [0.8, 0.9, 0.7]},
            # },
            'sigma_t': {
                'type': 'gridvolume',
                'grid': grids[0],
                'to_world': to_world,
            },
            'emission': {
                'type': 'gridvolume',
                'grid': grids[1],
                'to_world': to_world,
            },
            'albedo': {
                'type': 'gridvolume',
                'grid': grids[2],
                'to_world': to_world,
            },
        },
        # -------------------- Shapes --------------------
        'cube': {
            # Cube covers [0, 0, 0] to [1, 1, 1] by default
            'type': 'obj',
            'filename': join(SCENE_DIR, 'common', 'meshes', 'cube_unit.obj'),
            'bsdf': { 'type': 'null', },
            'interior': {
                'type': 'ref',
                'id':  'medium1'
            },
            'to_world': to_world,
        },
    }


def loss_fn(image):
    return dr.mean(dr.sqr(image - 0.5))


def test_01_nerf_basic():
    output_dir = join(OUTPUT_DIR, 'test_integrators', 'test_nerf_basic')
    os.makedirs(output_dir, exist_ok=True)
    mi.set_variant('cuda_ad_rgb')
    from integrators.nerf import NeRFIntegrator

    integrator = mi.load_dict({
        'type': 'nerf',
        'queries_per_ray': 64,
        'activation': 'relu',
        'hide_emitters': False,
    })

    scene = mi.load_file(join(SCENE_DIR, 'janga-smoke', 'janga-smoke.xml'),
                         pixel_format='RGBA', resx=128, resy=128, spp=4)

    params = mi.traverse(scene)
    params.keep(['medium1.sigma_t.data', 'medium1.emission.data'])
    assert len(params) == 2
    print(params)

    for k, p in params.items():
        dr.enable_grad(p)

    img = mi.render(scene, integrator=integrator, seed=1234, params=params)
    mi.Bitmap(img).write(join(output_dir, 'primal.exr'))

    loss = loss_fn(img)
    dr.backward(loss)

    for k, p in params.items():
        g = dr.grad(p)
        print(k, dr.sum(g)[0], g)


def test_02_nerf_correctness():
    output_dir = join(OUTPUT_DIR, 'test_integrators', 'test_nerf_correctness')
    os.makedirs(output_dir, exist_ok=True)
    mi.set_variant('cuda_ad_rgb')
    from integrators.nerf import NeRFIntegrator

    scene_dict = cube_test_scene()
    scene = mi.load_dict(scene_dict)
    integrator = mi.load_dict({ 'type': 'nerf', })

    params = mi.traverse(scene)
    params.keep(['cube.interior_medium.sigma_t.data', 'cube.interior_medium.emission.data'])
    assert len(params) == 2

    fd_cache = join(output_dir, 'fd.pickle')
    @pickle_cache(fd_cache, overwrite=False)
    def get_fd_grads():
        return fd_gradients(output_dir, scene, params, loss_fn, eps=5e-3, spp=32,
                            write_images=True, integrator=integrator)

    rb_cache = join(output_dir, 'rb.pickle')
    @pickle_cache(rb_cache, overwrite=True)
    def get_rb_grads():
        for k, v in params.items():
            dr.enable_grad(v)
        # Note: too many spp will lead to precision error (accumulation)
        img = mi.render(scene, integrator=integrator, seed=1234, params=params, spp=4)
        mi.Bitmap(img).write(join(output_dir, 'nerf_primal.exr'))
        loss = loss_fn(img)
        dr.backward(loss)
        return { k: dr.grad(v).numpy() for k, v in params.items() }

    # Test closeness of gradients
    # The automated threshold is quite relaxed and allows for a
    # a few entries disagreeing relatively significantly.
    results = {}
    results['fd'] = get_fd_grads()
    results['rb'] = get_rb_grads()
    ref_method = 'fd'
    rtol = 3e-2
    for method, grads in results.items():
        for k, g in grads.items():
            for c in range(g.shape[-1]):
                img = gallery(g[..., c][..., None])
                fname = join(output_dir, f'grads_{k}_{c}_{method}.exr')
                mi.Bitmap(img.astype(np.float32)).write(fname)
                print(f'[+] {fname}')

                if method != ref_method:
                    a = g[..., c]
                    b = results[ref_method][k][..., c]
                    bad = np.sum(np.abs(a - b) >= rtol * np.abs(b))
                    if bad > 0:
                        print(a)
                        print('vs')
                        print(b)
                        print('relative:')
                        print(np.abs(a - b) / np.abs(b))
                    assert bad <= 3, (method, k, c, bad)
                    # Upper bound even on bad entries (very tolerant)
                    assert np.allclose(a, b, rtol=0.75), (method, k, c, bad)



def test_03_volpathsimple_basic():
    output_dir = join(OUTPUT_DIR, 'test_integrators', 'test_volpathsimple_basic')
    os.makedirs(output_dir, exist_ok=True)
    mi.set_variant('cuda_ad_rgb')
    from integrators.volpathsimple import VolpathSimpleIntegrator

    scene_dict = cube_test_scene(density_scale=2.0)
    scene = mi.load_dict(scene_dict)
    integrators = {
        'volpath': mi.load_dict({
            'type': 'volpath',
            'max_depth': 64,
            'rr_depth': 999,
        }),
        'volpathsimple':  mi.load_dict({
            'type': 'volpathsimple',
            'max_depth': 64,
            'use_nee': True,
            'rr_depth': 999,
        }),
    }

    results = {}
    for k, integrator in integrators.items():
        fname = join(output_dir, f'preview_{k}.exr')
        @render_cache(fname, overwrite=False)
        def fn():
            return mi.render(scene, integrator=integrator, spp=4096)
        results[k] = fn()

    ref_integrator = 'volpath'
    for k, image in results.items():
        if k != ref_integrator:
            a = image
            b = results[ref_integrator]
            assert np.allclose(a, b, atol=5e-2)



def test_04_volpathsimple_correctness():
    mi.set_variant('cuda_ad_rgb_double')
    from integrators.volpathsimple import VolpathSimpleIntegrator

    scene_dict = cube_test_scene(density_scale=2.0)
    scene = mi.load_dict(scene_dict)
    integrator = mi.load_dict({
        'type': 'volpathsimple',
        # TODO
        # 'max_depth': 1,
        'max_depth': 64,
        'rr_depth': 999,
        # TODO: test both
        'use_drt': False,
        # TODO: test both
        'use_nee': True,
    })
    seed = 12345

    suffix = ''
    if integrator.use_nee:
        suffix += '_nee'
    if integrator.use_drt:
        suffix += '_drt'

    output_dir = join(OUTPUT_DIR, 'test_integrators', f'test_volpathsimple_correctness{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    params = mi.traverse(scene)
    params.keep(['cube.interior_medium.sigma_t.data', 'cube.interior_medium.albedo.data'])
    assert len(params) == 2

    fd_cache = join(output_dir, 'fd.pickle')
    @pickle_cache(fd_cache, overwrite=False)
    def get_fd_grads():
        return fd_gradients(output_dir, scene, params, loss_fn, eps=5e-3, spp=4096,
                            write_images=True, integrator=integrator, seed=seed)

    volpathsimple_cache = join(output_dir, 'volpathsimple.pickle')
    @pickle_cache(volpathsimple_cache, overwrite=True)
    def get_volpathsimple_grads():
        for k, v in params.items():
            dr.enable_grad(v)
        # Note: too many spp will lead to precision error (accumulation)
        img = mi.render(scene, integrator=integrator, params=params,
                        spp=512, seed=seed)
        mi.Bitmap(img) \
          .convert(component_format=mi.Struct.Type.Float32) \
          .write(join(output_dir, 'volpathsimple_primal.exr'))
        loss = loss_fn(img)
        dr.backward(loss)
        return { k: dr.grad(v).numpy() for k, v in params.items() }

    # Test closeness of gradients
    # The automated threshold is quite relaxed and allows for a
    # a few entries disagreeing relatively significantly.
    results = {}
    results['fd'] = get_fd_grads()
    results['volpathsimple'] = get_volpathsimple_grads()
    ref_method = 'fd'
    rtol = 3e-2
    for method, grads in results.items():
        for k, g in grads.items():
            for c in range(g.shape[-1]):
                img = gallery(g[..., c][..., None])
                fname = join(output_dir, f'grads_{k}_{c}_{method}.exr')
                mi.Bitmap(img.astype(np.float32)).write(fname)
                print(f'[+] {fname}')

                if method != ref_method:
                    a = g[..., c]
                    b = results[ref_method][k][..., c]
                    bad = np.sum(np.abs(a - b) >= rtol * np.abs(b))
                    if bad > 0:
                        print(a)
                        print('vs')
                        print(b)
                        print('relative:')
                        print(np.abs(a - b) / np.abs(b))
                    # TODO: re-enable this
                    if False:
                        assert bad <= 3, (method, k, c, bad)
                        # Upper bound even on bad entries (very tolerant)
                        assert np.allclose(a, b, rtol=0.75), (method, k, c, bad)




if __name__ == '__main__':
    # test_02_nerf_correctness()
    # test_03_volpathsimple_basic()
    test_04_volpathsimple_correctness()
