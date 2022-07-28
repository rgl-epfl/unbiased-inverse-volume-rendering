from dataclasses import fields
import os
from os.path import join
from copy import deepcopy

import drjit as dr
import mitsuba as mi
from tqdm import tqdm

from batched import render_batch
from util import save_params, get_single_medium


def load_scene(scene_config, reference=False, **kwargs):
    scene_vars = scene_config.ref_scene_vars if reference else scene_config.normal_scene_vars
    if reference and scene_config.ref_fname:
        scene = mi.load_file(scene_config.ref_fname, **scene_vars, **kwargs)
    else:
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


def load_reference_images(paths, batchify=False):
    if batchify:
        import numpy as np
        # Note: we rely on `paths` being ordered consistently.
        batched = np.concatenate([
            np.array(mi.Bitmap(f))[None, ...]
            for _, f in paths.items()
        ], axis=0)
        return mi.TensorXf(batched)
    else:
        return {
            s: mi.TensorXf(mi.Bitmap(f))
            for s, f in paths.items()
        }


def gather_ref_values(ref_images, sensor_idx, pixel_idx):
    # Shape: image_idx * height * width * channels
    sh = dr.shape(ref_images)
    assert len(sh) == 4
    channels = sh[-1]
    assert channels in (3, 4)
    sh = sh[:3]
    indices = (
        sensor_idx * dr.prod(sh[1:])
        + pixel_idx.y * dr.prod(sh[2:])
        + pixel_idx.x * dr.prod(sh[3:])
    )
    color_type = mi.Vector3f if channels == 3 else mi.Vector4f

    values = dr.gather(color_type, ref_images.array, indices)

    # Shape: height * batch_size * channels
    return mi.TensorXf(dr.ravel(values), shape=(1, dr.width(pixel_idx), channels))


def render_previews(output_dir, opt_config, scene_config, scene, integrator, it_i):
    if it_i == 'initial':
        if not opt_config.render_initial:
            return
        suffix = '_init'
    elif it_i == 'final':
        if not opt_config.render_final:
            return
        suffix = '_final'
    elif isinstance(it_i, int):
        suffix = f'_{it_i:08d}'
    else:
        assert isinstance(it_i, str)
        suffix = it_i

    preview_spp = opt_config.preview_spp or opt_config.spp

    for s in scene_config.preview_sensors:
        fname = join(output_dir, f'opt{suffix}_{s:04d}.exr')
        image = mi.render(scene, integrator=integrator, sensor=s,
                          seed=1234, spp=preview_spp,)
        mi.Bitmap(image).write(fname)


def initialize_scene(opt_config, scene_config, scene):
    params = mi.traverse(scene)
    params.keep(scene_config.param_keys)

    # Set params to their initial values
    for k, v in scene_config.start_from_value.items():
        assert k in params
        if v is None:
            assert not opt_config.upsample
            continue

        # --- Upsampling support
        # If parameter resolutions will be upsampled during the optimization,
        # figure out the initial resolution that will lead to the correct
        # final resolution after n upsampling steps.
        shape = dr.shape(params[k])
        if opt_config.upsample:
            assert len(shape) == 4
            upsample_res_factor = 2 ** len(opt_config.upsample)
            # Preserve channel count
            init_res = (*[max(1, s // upsample_res_factor) for s in shape[:3]], shape[-1])
            if 1 in init_res[:3]:
                raise ValueError(f'Initial resolution not supported: {init_res}. Maybe reduce upsample_steps?')

            if '.sigma_t.' in k:
                adjust_majorant_res_factor(scene_config, scene, init_res)
        else:
            init_res = shape

        params[k] = type(params[k])(v, shape=init_res)

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


def adjust_majorant_res_factor(scene_config, scene, density_res):
    res_factor = scene_config.majorant_resolution_factor

    if res_factor > 1:
        min_side = dr.min(density_res[:3])
        # For the current density res, find the largest factor that
        # results in a meaningful supergrid resolution.
        while (res_factor > 1) and (min_side // res_factor) < 4:
            res_factor -= 1
    # Otherwise, just disable the supergrid.
    if res_factor <= 1:
        res_factor = 0

    medium = get_single_medium(scene)
    current = medium.majorant_resolution_factor()
    if current != res_factor:
        medium.set_majorant_resolution_factor(res_factor)
        print(f'[i] Updated majorant supergrid resolution factor: {current} → {res_factor}')



def upsample_grid(current_values, old_res, new_res, n_channels):
    """Upsample values of a 3D grid using first order interpolation."""
    from scipy.ndimage import zoom
    assert isinstance(current_values, (mi.TensorXf, dr.detached_t(mi.TensorXf))), \
            'Unsupported type for upsampling: {}'.format(type(current_values))
    if tuple(old_res) == tuple(new_res):
        return mi.TensorXf(dr.detach(current_values))

    n_channels = old_res[-1]
    assert len(old_res) == 4 and len(new_res) == 4
    assert new_res[-1] == n_channels

    # TODO: replace with a pure DrJit-based solution
    factors = [r / old_res for r in new_res]
    current_values = current_values.numpy()
    if current_values.ndim == 3:
        current_values = current_values[..., None]

    new_values = zoom(current_values, factors, order=1,
                      mode='nearest', prefilter=False, grid_mode=True)
    new_values = mi.TensorXf(new_values)
    assert new_values.shape == new_res
    return new_values


def upsample_params_if_needed(opt_config, scene_config, scene, params, opt, it_i):
    if not opt_config.should_upsample(it_i):
        return False

    majorant_res_factor = scene_config.majorant_resolution_factor

    for k in scene_config.param_keys:
        v = opt[k]
        old_res = dr.shape(v)
        assert len(old_res) == 4
        new_res = (*[2 * r for r in old_res[:3]], old_res[-1])

        # opt[k] = dr.upsample(v, shape=new_res)
        opt[k] = upsample_grid(v, old_res, new_res)

        assert dr.shape(opt[k]) == new_res
        print(f'[i] Upsampled parameter "{k}" at iteration {it_i}: {old_res} → {new_res}')

        if '.sigma_t.' in k:
            adjust_majorant_res_factor(scene_config, scene, new_res)

    medium = get_single_medium(scene)
    medium.set_majorant_resolution_factor(majorant_res_factor)
    params.update(opt)
    return True


def create_checkpoint(output_dir, opt_config, scene_config, params, name_or_it):
    prefix = name_or_it
    if name_or_it == 'initial':
        if not opt_config.checkpoint_initial:
            return
    elif name_or_it == 'final':
        if not opt_config.checkpoint_final:
            return
    elif isinstance(name_or_it, int):
        if (name_or_it == 0) or (not opt_config.checkpoint_stride) or (name_or_it % opt_config.checkpoint_stride) != 0:
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


    batch_size = opt_config.batch_size
    ref_paths = get_reference_image_paths(scene_config)
    ref_images = load_reference_images(ref_paths, batchify=(batch_size is not None))
    scene = load_scene(scene_config, reference=False)
    integrator = int_config.create(max_depth=scene_config.max_depth)
    sampler = mi.scalar_rgb.PCG32(initstate=93483)

    n_sensors = len(scene_config.sensors)
    spp_grad = opt_config.spp
    spp_primal = spp_grad * opt_config.primal_spp_factor

    if batch_size is not None:
        sensors_dr = dr.gather(mi.SensorPtr, scene.sensors_dr(),
                               mi.UInt32(scene_config.sensors))
        # Assume that all sensors have the same dimensions
        first_film = scene.sensors()[scene_config.sensors[0]].film()
        film_size = first_film.crop_size()
        pixel_format = (
            mi.Bitmap.PixelFormat.RGBA if mi.has_flag(first_film.flags(), mi.FilmFlags.Alpha)
            else mi.Bitmap.PixelFormat.RGB
        )
        batch_film = None
        batch_render_sampler = None
        del first_film

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
    for it_i in tqdm(range(opt_config.n_iter), desc='Optimization',
                     dynamic_ncols=True):
        seed, _ = mi.sample_tea_32(2 * it_i + 0, opt_config.base_seed)
        seed_grad, _ = mi.sample_tea_32(2 * it_i + 1, opt_config.base_seed)
        opt.set_learning_rate(opt_config.learning_rates(scene_config, it_i))
        upsample_params_if_needed(opt_config, scene_config, scene, params, opt, it_i)

        if batch_size is not None:
            # --- Batched rendering
            image, batch_film, batch_render_sampler, sensor_idx, pixel_idx = render_batch(
                batch_size, scene, sensors_dr, film_size,
                params=params, integrator=integrator, film=batch_film,
                pixel_format=pixel_format, sampler=batch_render_sampler,
                spp=spp_primal, spp_grad=spp_grad,
                seed=seed, seed_grad=seed_grad
            )
            ref_values = gather_ref_values(ref_images, sensor_idx, pixel_idx)
        else:
            # --- Sensor-based rendering
            sensor_i = scene_config.sensors[int(sampler.next_float32() * n_sensors)]
            image = mi.render(scene, params=params, integrator=integrator, sensor=sensor_i,
                              spp=spp_primal, spp_grad=spp_grad,
                              seed=seed, seed_grad=seed_grad)
            ref_values = ref_images[sensor_i]
        loss_value = opt_config.loss(image, ref_values)
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
