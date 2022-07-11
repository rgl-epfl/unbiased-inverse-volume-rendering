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

def test_01_nerf_integrator():
    mi.set_variant('cuda_ad_rgb')
    from integrators.nerf import NeRFIntegrator

    integrator = mi.load_dict({
        'type': 'nerf',
        'queries_per_ray': 64,
        'activation': 'relu',
        'hide_emitters': False,
    })

    scene = mi.load_file(join(SCENE_DIR, 'janga-smoke', 'janga-smoke.xml'),
                         pixel_format='RGBA', resx=128, resy=128)
    # scene = mi.load_dict({
    #     'type': 'scene',
    #     'rectangle': {
    #         'type': 'rectangle',
    #     },
    #     'emitter': {
    #         'type': 'constant',
    #     },
    # })

    params = mi.traverse(scene)
    params.keep(['medium1.sigma_t.data', 'medium1.emission.data'])
    assert len(params) == 2
    print(params)

    for k, p in params.items():
        dr.enable_grad(p)

    img = mi.render(scene, integrator=integrator, seed=1234, params=params)
    mi.Bitmap(img).write(join(OUTPUT_DIR, 'primal.exr'))

    loss = dr.sqr(dr.sum(img) - 0.5)
    dr.backward(loss)

    for k, p in params.items():
        g = dr.grad(p)
        print(k, dr.sum(g)[0], g)


if __name__ == '__main__':
    test_01_nerf_integrator()
