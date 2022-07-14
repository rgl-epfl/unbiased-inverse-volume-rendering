
from dataclasses import fields
from multiprocessing.sharedctypes import Value
import os
from os.path import join
from copy import deepcopy

import drjit as dr
import mitsuba as mi
from tqdm import tqdm

from constants import OUTPUT_DIR
from scene_config import get_scene_config
from util import save_params


def load_scene(scene_config, reference=False, **kwargs):
    scene_vars = scene_config.ref_scene_vars if reference else scene_config.normal_scene_vars
    scene = mi.load_file(scene_config.fname, **scene_vars, **kwargs)

    return scene


def render_reference_image(scene_config, to_render, seed=1234, max_rays_per_pass=720*720*2048):
    import integrators
    scene = load_scene(scene_config, reference=True)
    integrator = mi.load_dict({
        'type': scene_config.ref_integrator,
        'max_depth': scene_config.max_depth,
    })
    ref_spp = scene_config.ref_spp

    for s, fname in tqdm(to_render.items(), desc=f'Reference renderings @ {ref_spp}'):
        sensor = scene.sensors()[s]
        # Assuming we're rendering on the GPU, split up the renderings to avoid
        # running out of memory.
        total_rays = dr.prod(sensor.film().size()) * ref_spp
        pass_count = int(dr.ceil(total_rays / max_rays_per_pass))
        spp_per_pass = int(dr.ceil(ref_spp / pass_count))
        assert spp_per_pass * pass_count >= ref_spp

        result = None
        for pass_i in tqdm(range(pass_count), desc='Render passes', leave=False):
            image = mi.render(scene, sensor=sensor, integrator=integrator,
                              spp=spp_per_pass, seed=seed + pass_i)
            dr.eval(image)
            if result is None:
                result = image / pass_count
            else:
                result += image / pass_count
            del image

        mi.Bitmap(result).write(fname)


def get_reference_image_paths(scene_config, overwrite=False):
    ref_dir = scene_config.references
    os.makedirs(ref_dir, exist_ok=True)

    fname_pattern = join(ref_dir, 'ref_{:06d}.exr')
    paths = { s: fname_pattern.format(s) for s in scene_config.sensors }

    # Render reference images if needed
    if overwrite:
        missing_paths = deepcopy(paths)
    else:
        missing_paths = { s: fname for s, fname in paths.items()
                          if not os.path.isfile(fname) }
    if missing_paths:
        render_reference_image(scene_config, missing_paths)
    return paths


def load_reference_images(paths):
    return {
        s: mi.TensorXf(mi.Bitmap(f))
        for s, f in paths.items()
    }


def render_previews(output_dir, opt_config, scene_config, scene, integrator, it_i):
    if it_i == 'initial':
        if not opt_config.render_initial:
            return
        suffix = '_init'
    elif it_i == 'final':
        if not opt_config.render_final:
            return
        suffix = '_final'
    else:
        suffix = f'_{it_i:08d}'

    preview_spp = opt_config.preview_spp or opt_config.spp

    for s in scene_config.preview_sensors:
        fname = join(output_dir, f'opt{suffix}_{s:04d}.exr')
        image = mi.render(scene, integrator=integrator, sensor=s,
                          seed=1234, spp=preview_spp,)
        mi.Bitmap(image).write(fname)


def initialize_scene(opt_config, scene_config, scene):
    params = mi.traverse(scene)
    params.keep(scene_config.param_keys)
    # TODO: downsample params to their initial resolution

    # Set params to their initial values
    for k, v in scene_config.start_from_value.items():
        assert k in params
        params[k] = type(params[k])(v, shape=dr.shape(params[k]))
    params.update()
    return params


def enforce_valid_params(scene_config, opt):
    """Projects parameters back to their legal range."""
    for k, v in opt.items():
        if k.endswith('sigma_t.data'):
            opt[k] = dr.clip(v, 0, scene_config.max_density)
        elif k.endswith('emission.data'):
            opt[k] = dr.maximum(v, 0)
        elif k.endswith('albedo.data'):
            opt[k] = dr.clip(v, 0, 1)
        else:
            raise ValueError


def create_checkpoint(output_dir, opt_config, scene_config, params, name_or_it):
    prefix = name_or_it
    if name_or_it == 'initial':
        if not opt_config.checkpoint_initial:
            return
    elif name_or_it == 'final':
        if not opt_config.checkpoint_final:
            return
    elif isinstance(name_or_it, int):
        if (name_or_it == 0) or (name_or_it % opt_config.checkpoint_stride) != 0:
            return
        prefix = f'{name_or_it:08d}'
    else:
        raise ValueError('Unsupported: ' + str(name_or_it))

    checkpoint_dir = join(output_dir, 'params')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_params(checkpoint_dir, scene_config, params, prefix)


def run_optimization(output_dir, opt_config, scene_config, int_config):
    import integrators
    print(f'[i] Starting optimization:')
    print(f'    Scene:      {scene_config.name}')
    print(f'    Integrator: {int_config.name}')
    print(f'    Output dir: {output_dir}')
    print(f'    Opt params:')
    for f in fields(opt_config):
        print(f'        {f.name}: {opt_config.__dict__[f.name]}')


    ref_paths = get_reference_image_paths(scene_config)
    ref_images = load_reference_images(ref_paths)
    scene = load_scene(scene_config, reference=False)
    integrator = int_config.create(max_depth=scene_config.max_depth)
    sampler = mi.scalar_rgb.PCG32(initstate=93483)

    n_sensors = len(scene_config.sensors)
    spp_grad = opt_config.spp
    spp_primal = spp_grad * opt_config.primal_spp_factor

    # --- Initialization
    params = initialize_scene(opt_config, scene_config, scene)
    opt = opt_config.optimizer(params)
    for _, v in params.items():
        dr.enable_grad(v)

    create_checkpoint(output_dir, opt_config, scene_config, params, 'initial')
    render_previews(output_dir, opt_config, scene_config, scene, integrator, 'initial')
    # Write out the reference images corresponding to the previews for easy comparison
    for s in scene_config.preview_sensors:
        fname = join(output_dir, f'ref_{s:04d}.exr')
        mi.Bitmap(ref_images[s]).write(fname)

    # --- Main optimization loop
    for it_i in tqdm(range(opt_config.n_iter), desc='Optimization'):
        seed, _ = mi.sample_tea_32(2 * it_i + 0, opt_config.base_seed)
        seed_grad, _ = mi.sample_tea_32(2 * it_i + 1, opt_config.base_seed)
        # TODO: LR schedule

        # TODO: support batched rendering
        if opt_config.batch_size is not None:
            raise NotImplementedError('Batched rendering')

        sensor_i = scene_config.sensors[int(sampler.next_float32() * n_sensors)]
        # TODO: upsampling of params

        image = mi.render(scene, params=params, integrator=integrator, sensor=sensor_i,
                          spp=spp_primal, spp_grad=spp_grad,
                          seed=seed, seed_grad=seed_grad)
        loss_value = opt_config.loss(image, ref_images[sensor_i])
        dr.backward(loss_value)

        opt.step()
        enforce_valid_params(scene_config, opt)
        params.update(opt)
        create_checkpoint(output_dir, opt_config, scene_config, params, it_i)

        if (it_i > 0) and (it_i % opt_config.preview_stride) == 0:
            render_previews(output_dir, opt_config, scene_config, scene, integrator, it_i)
    # ------

    create_checkpoint(output_dir, opt_config, scene_config, params, 'final')
    render_previews(output_dir, opt_config, scene_config, scene, integrator, 'final')
    print(f'[✔︎] Optimization complete: {opt_config.name}\n')

    return scene, params, opt
