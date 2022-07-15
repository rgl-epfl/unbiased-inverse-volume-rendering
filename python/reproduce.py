"""
Reproduces results from the article:
    Merlin Nimier-David, Thomas Müller, Alexander Keller, and Wenzel Jakob. 2022.
    Unbiased Inverse Volume Rendering with Differential Trackers.
    In Transactions on Graphics (Proceedings of SIGGRAPH) 41(4).
"""

from copy import deepcopy
import os
from os.path import join

from pyrsistent import get_in

import drjit as dr
import mitsuba as mi
from tqdm import tqdm

from constants import OUTPUT_DIR
from opt_config import get_int_config, OptimizationConfig, Schedule
from optimize import run_optimization
from scene_config import get_scene_config


def reproduce_optimization_experiments(configs, overwrite=False):
    for cname, entries in configs.items():
        exp_output_dir = join(OUTPUT_DIR, cname)
        scene_config = get_scene_config(entries['scene'])

        for int_name, opt_overrides in entries['integrators'].items():
            int_config = get_int_config(int_name)
            opt_config = deepcopy(entries['opt'])
            if opt_overrides:
                opt_config.update(opt_overrides)
            opt_config = OptimizationConfig(name=cname, **opt_config)

            output_dir = join(exp_output_dir, int_name)
            os.makedirs(output_dir, exist_ok=True)

            result_fname = join(output_dir, 'params', 'final-medium1_sigma_t.vol')
            if overwrite or not os.path.isfile(result_fname):
                run_optimization(output_dir, opt_config, scene_config, int_config)


def main():
    base_opt_config = {
        'n_iter': 1000,
        'spp': 16,
        'primal_spp_factor': 64,
        'lr': 5e-3,
        'lr_schedule': Schedule.Last25,
        'batch_size': None,
        'render_initial': True,
        'render_final': True,
        'preview_stride': 100

        # 'n_iter': 6000,
        # 'spp': 16,
        # 'primal_spp_factor': 64,
        # 'lr': 5e-3,
        # 'batch_size': 32768,
        # 'render_initial': False,
        # 'render_final': False,
    }

    configs = {
        'janga-smoke-sn64': {
            'scene': 'janga-smoke',
            'opt': deepcopy(base_opt_config),
            # Integrator name => optimization config overrides
            'integrators': {
                'nerf': {
                    'lr': 1e-2,
                    'spp': 4,
                    'primal_spp_factor': 1,
                },
                # 'volpathsimple-basic': None,
                # 'volpathsimple-drt': None,
                # TODO: support NeRF initiialization
            },
        },
    }
    reproduce_optimization_experiments(configs)



if __name__ == '__main__':
    mi.set_variant('cuda_ad_rgb')
    main()
