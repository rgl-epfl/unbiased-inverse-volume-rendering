"""
Support for batched rendering (batches or arbitrary rays),
as opposed to sensor-based rendering provided by `mi.render`.
"""
from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import gc

import drjit as dr
import mitsuba as mi


class _BatchedRenderOp(dr.CustomOp):
    """
    Batched (i.e. ray-centric) alternative to `mi._RenderOp`.
    """

    def eval(self, scene, sensors, sensors_idx, sampled_pixels, film_size, batch_samplers,
             params, integrator, film, pixel_format, sampler, seed, spp):
        self.scene = scene
        self.sensors_idx = sensors_idx
        self.sampled_sensors = dr.gather(mi.SensorPtr, sensors, sensors_idx)
        self.sampled_pixels = sampled_pixels
        self.film_size = film_size
        self.batch_samplers = batch_samplers
        self.params = params
        self.integrator = integrator
        self.film = film
        self.pixel_format = pixel_format
        self.sampler = sampler
        self.seed = seed
        self.spp = spp

        with dr.suspend_grad():
            # Sample a set of rays for primal rendering
            rays, ray_weight, pos = sample_batch_rays(
                self.sampled_sensors, self.sampled_pixels,
                self.film_size, self.batch_samplers[1], self.spp[0]
            )
            # TODO: support ray_weight != 1

            image, film, render_sampler = render_batch_primal(
                integrator=self.integrator,
                film=self.film,
                pixel_format=self.pixel_format,
                sampler=self.sampler,
                scene=self.scene,
                rays=rays,
                seed=seed[0],
                spp=spp[0],
                develop=True,
                evaluate=False
            )
            return (image, film, render_sampler,
                    dr.detach(self.sensors_idx, preserve_type=False),
                    dr.detach(self.sampled_pixels, preserve_type=False))

    def forward(self):
        if not isinstance(self.params, mi.SceneParameters):
            raise Exception('An instance of mi.SceneParameter containing the '
                            'scene parameter to be differentiated should be '
                            'provided to mi.render() if forward derivatives are '
                            'desired!')
        self.set_grad_out(
            render_batch_forward(self.integrator, self.film, self.sampler, self.scene,
                                 self.params, self.rays, self.seed[1], self.spp[1])
        )

    def backward(self):
        # Sample a separate, decorrelated set of rays for the adjoint (but passing
        # through the same set of pixels).
        rays, ray_weight, pos = sample_batch_rays(
            self.sampled_sensors, self.sampled_pixels,
            self.film_size, self.batch_samplers[2], self.spp[1]
        )
        # TODO: support ray_weight != 1

        image_grad_in = self.grad_out()[0]
        render_batch_backward(self.integrator, self.film, self.sampler, self.scene,
                              self.params, image_grad_in, rays,
                              seed=self.seed[1], spp=self.spp[1],
                              pixel_format=self.pixel_format)

    def name(self):
        return "BatchedRenderOp"


def render_batch(batch_size: int,
                 scene: mi.Scene,
                 sensors: mi.SensorPtr,
                 film_size: mi.ScalarVector2u,
                 params: Any = None,
                 integrator: mi.Integrator = None,
                 film: mi.Film = None,
                 pixel_format: mi.Bitmap.PixelFormat = None,
                 sampler: mi.Sampler = None,
                 seed: int = 0,
                 seed_grad: int = 0,
                 spp: int = 0,
                 spp_grad: int = 0):
    """
    Batched (i.e. ray-centric) alternative to the sensor-centric `mi.render`.
    """

    if params is not None and not isinstance(params, mi.SceneParameters):
        raise Exception('params should be an instance of mi.SceneParameter!')

    assert isinstance(scene, mi.Scene)

    if integrator is None:
        integrator = scene.integrator()
    assert isinstance(integrator, mi.Integrator)
    if pixel_format is None:
        pixel_format = mi.Bitmap.PixelFormat.RGB

    if spp_grad == 0:
        spp_grad = spp

    if seed_grad == 0:
        # Compute a seed that de-correlates the primal and differential phase
        seed_grad = mi.sample_tea_32(seed, 1)[0]
    elif seed_grad == seed:
        raise Exception('The primal and differential seed should be different '
                        'to ensure unbiased gradient computation!')

    sensors_idx, sampled_pixels, batch_samplers = sample_batch_pixels(
        batch_size, spp, spp_grad, sensors, film_size, seed)

    return dr.custom(_BatchedRenderOp, scene, sensors, sensors_idx, sampled_pixels, film_size,
                     batch_samplers, params, integrator, film, pixel_format, sampler,
                     (seed, seed_grad), (spp, spp_grad))


def render_batch_primal(integrator: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        rays: mi.RayDifferential3f,
                        seed: int = 0,
                        spp: int = 0,
                        develop: bool = True,
                        evaluate: bool = True,
                        film: mi.Film = None,
                        pixel_format: mi.Bitmap.PixelFormat = None,
                        sampler: mi.Sampler = None) -> mi.TensorXf:

    if not develop:
        raise Exception("develop=True must be specified when "
                        "invoking AD integrators")

    # Disable derivatives in all of the following
    with dr.suspend_grad():
        # Prepare the film and sample generator for rendering
        film, sampler, spp = prepare_batch(rays, seed, spp=spp, aovs=integrator.aovs(),
                                           film=film, pixel_format=pixel_format,
                                           sampler=sampler)

        # Assume that rays were perfectly sampled.
        ray_weight = 1.0
        pos = mi.Point2f(
            0.5 + mi.Float(dr.arange(mi.UInt32, dr.width(rays)) // spp),
            0.5
        )
        # Launch the Monte Carlo sampling process in primal mode
        L, valid, state = integrator.sample(
            mode=dr.ADMode.Primal,
            scene=scene,
            sampler=sampler,
            ray=rays,
            depth=mi.UInt32(0),
            δL=None,
            state_in=None,
            reparam=None,
            active=mi.Bool(True)
        )

        # Prepare an ImageBlock as specified by the film
        block = film.create_block()

        # Only use the coalescing feature when rendering enough samples
        block.set_coalesce(block.coalesce() and spp >= 4)

        # Accumulate into the image block
        alpha = dr.select(valid, mi.Float(1), mi.Float(0))
        if mi.has_flag(film.flags(), mi.FilmFlags.Special):
            aovs = film.prepare_sample(L * ray_weight, rays.wavelengths,
                                       block.channel_count(), alpha=alpha)
            block.put(pos, aovs)
            del aovs
        else:
            block.put(pos, rays.wavelengths, L * ray_weight, alpha)

        # Explicitly delete any remaining unused variables
        del rays, ray_weight, pos, L, valid, alpha
        gc.collect()

        # Perform the weight division and return an image tensor
        film.put_block(block)
        return film.develop(), film, sampler


def render_batch_forward(integrator: mi.SamplingIntegrator,
                         film: mi.Film,
                         sampler: mi.Sampler,
                         scene: mi.Scene,
                         params: Any,
                         rays: mi.RayDifferential3f,
                         seed: int = 0,
                         spp: int = 0,
                         pixel_format: mi.Bitmap.PixelFormat = None) -> mi.TensorXf:
    raise NotImplementedError('BatchedRenderOp with forward gradient propagation.')


def render_batch_backward(integrator: mi.SamplingIntegrator,
                          film: mi.Film,
                          sampler: mi.Sampler,
                          scene: mi.Scene,
                          params: Any,
                          grad_in: mi.TensorXf,
                          rays: mi.RayDifferential3f,
                          seed: int = 0,
                          spp: int = 0,
                          pixel_format: mi.Bitmap.PixelFormat = None) -> None:

    aovs = integrator.aovs()

    # Disable derivatives in all of the following, except where explicitly re-enabled
    with dr.suspend_grad():
        # Prepare the film and sample generator for rendering
        film, sampler, spp = prepare_batch(rays, seed, spp, aovs,
                                           film=film, sampler=sampler,
                                           pixel_format=pixel_format)

        # When the underlying integrator supports reparameterizations,
        # perform necessary initialization steps and wrap the result using
        # the _ReparamWrapper abstraction defined above
        if hasattr(integrator, 'reparam'):
            reparam = mi.ad.common._ReparamWrapper(
                scene=scene,
                params=params,
                reparam=integrator.reparam,
                wavefront_size=sampler.wavefront_size(),
                seed=seed
            )
        else:
            reparam = None

        # Assume that rays were perfectly sampled.
        ray_weight = 1.0
        det = 1.0
        pos = mi.Point2f(
            0.5 + mi.Float(dr.arange(mi.UInt32, dr.width(rays)) // spp),
            0.5
        )

        # (1) Primal rendering (detached)
        L, valid, state_out = integrator.sample(
            mode=dr.ADMode.Primal,
            scene=scene,
            sampler=sampler.clone(),
            ray=rays,
            δL=None,
            state_in=None,
            active=mi.Bool(True),
            reparam=None,
        )

        # Prepare an ImageBlock as specified by the film
        block = film.create_block()

        # Only use the coalescing feature when rendering enough samples
        block.set_coalesce(block.coalesce() and spp >= 4)

        with dr.resume_grad():
            dr.enable_grad(L)

            # Accumulate into the image block
            if mi.has_flag(film.flags(), mi.FilmFlags.Special):
                aovs = film.prepare_sample(L * ray_weight * det, rays.wavelengths,
                                           block.channel_count(),
                                           weight=det,
                                           alpha=dr.select(valid, mi.Float(1), mi.Float(0)))
                block.put(pos, aovs)
                del aovs
            else:
                block.put(
                    pos=pos,
                    wavelengths=rays.wavelengths,
                    value=L * ray_weight * det,
                    weight=det,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0))
                )

            film.put_block(block)

            del valid
            gc.collect()

            # This step launches a kernel
            dr.schedule(state_out, block.tensor())
            image = film.develop()

            # Differentiate sample splatting and weight division steps to
            # retrieve the adjoint radiance
            dr.set_grad(image, grad_in)
            dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(mi.Float, dr.ADMode.Backward)
            δL = dr.grad(L)

        # (2) Launch Monte Carlo sampling in backward AD mode
        L_2, valid_2, state_out_2 = integrator.sample(
            mode=dr.ADMode.Backward,
            scene=scene,
            sampler=sampler,
            ray=rays,
            δL=δL,
            state_in=state_out,
            active=mi.Bool(True),
            reparam=reparam,
        )

        # We don't need any of the outputs here
        del L_2, valid_2, state_out, state_out_2, δL, \
            rays, ray_weight, pos, block, sampler
        gc.collect()

        # Run kernel representing side effects of the above
        dr.eval()


def prepare_batch(rays: mi.RayDifferential3f,
                  seed: int,
                  spp: int = 0,
                  aovs: list = [],
                  film: mi.Film = None,
                  pixel_format: mi.Bitmap.PixelFormat = None,
                  sampler: mi.Sampler = None):
    """
    Note that the only supported film reconstruction filter is `box`,
    since proximity of two rays in the wavefront order gives no
    information about their true proximity in sensor (image) space.
    """

    wavefront_size = dr.width(rays)
    assert (wavefront_size % spp) == 0
    n_channels, pixel_format_str = {
        mi.Bitmap.PixelFormat.RGB: (3, 'rgb'),
        mi.Bitmap.PixelFormat.RGBA: (4, 'rgba'),
        mi.Bitmap.PixelFormat.XYZ: (3, 'xyz'),
    }[pixel_format]
    n_channels += 1 + len(aovs)

    if film is None:
        assert pixel_format is not None
        film = mi.load_dict({
            'type': 'hdrfilm',
            'width': wavefront_size // spp,
            'height': 1,
            'pixel_format': pixel_format_str,
            'rfilter': {'type': 'box'},
        })
    else:
        # Ideally we would also check for the channel count and pixel format
        assert dr.all(film.crop_size() == mi.ScalarVector2u(wavefront_size // spp, 1))
        assert film.rfilter().is_box_filter()
        assert not film.sample_border()
    del pixel_format

    if sampler is None:
        sampler = mi.load_dict({'type': 'independent'})
    else:
        sampler = sampler.clone()

    if spp == 0:
        spp = sampler.sample_count()
    else:
        sampler.set_sample_count(spp)
    sampler.set_samples_per_wavefront(spp)

    is_llvm = dr.is_llvm_v(mi.Float)
    wavefront_size_limit = 0xffffffff if is_llvm else 0x40000000

    if wavefront_size >  wavefront_size_limit:
        raise Exception(
            "Tried to perform a %s-based rendering with a total sample "
            "count of %u, which exceeds 2^%u = %u (the upper limit "
            "for this backend). Please use fewer samples per pixel or "
            "render using multiple passes." %
            ("LLVM JIT" if is_llvm else "OptiX", wavefront_size,
                dr.log2i(wavefront_size_limit) + 1, wavefront_size_limit))

    sampler.seed(seed, wavefront_size)
    film.prepare(aovs)

    return film, sampler, spp



def sample_batch_pixels(batch_size: int,
                        spp: int,
                        spp_grad: int,
                        sensors: mi.MediumPtr,
                        film_size: mi.ScalarVector2u,
                        seed: int):
    n_sensors = dr.width(sensors)

    # We'll need samplers of three different sizes:
    # 1. One to sample individual pixels (`batch_size`)
    # 2. One to sample subpixels positions in the primal (`batch_size * spp`)
    # 3. One to sample subpixels positions in the adjoint (`batch_size * spp_grad`)
    batch_samplers = []
    for i, size in enumerate([batch_size, batch_size * spp, batch_size * spp_grad]):
        s = mi.load_dict({ 'type': 'independent', })
        sub_seed = mi.sample_tea_32(seed, 17 * i + 5)[0]
        s.seed(sub_seed, wavefront_size=size)
        batch_samplers.append(s)

    # First, pick a sensor for each ray
    sensor_idx = mi.UInt32(n_sensors * batch_samplers[0].next_1d())
    assert dr.width(sensor_idx) == batch_size

    # Sample individual pixels
    pixels = mi.Point2u(mi.Point2f(film_size) * batch_samplers[0].next_2d())

    return sensor_idx, pixels, batch_samplers


def sample_batch_rays(sampled_sensors, sampled_pixels, film_size, sampler, spp):
    """Assumes that all sensors have the same film dimensions."""
    batch_size = dr.width(sampled_pixels)

    # Extend wavefront from `batch_size` to `batch_size * spp`
    repeat_idx = dr.arange(mi.UInt32, batch_size * spp) // spp
    sensors = dr.gather(type(sampled_sensors), sampled_sensors, repeat_idx)
    pos = dr.gather(type(sampled_pixels), sampled_pixels, repeat_idx)
    del sampled_sensors, sampled_pixels

    # Subpixel positions
    offset = sampler.next_2d()
    assert dr.width(pos) == batch_size * spp
    assert dr.width(offset) == dr.width(pos)
    pos_f = mi.Vector2f(pos) + offset

    # Re-scale the position to [0, 1]^2
    pos_unit = dr.rcp(mi.ScalarVector2f(film_size)) * pos_f

    # TODO: support aperture sampling
    aperture_sample = mi.Vector2f(0.)
    # if sensor.needs_aperture_sample():
    #     aperture_sample = sampler.next_2d()

    # TODO: support time
    time = mi.Float(0.)
    # time = sensor.shutter_open()
    # if sensor.shutter_open_time() > 0:
    #     time += sampler.next_1d() * sensor.shutter_open_time()

    wavelength_sample = 0
    if mi.is_spectral:
        wavelength_sample = sampler.next_1d()

    # Finally, let each sensor sample rays from it (virtual function call)
    rays, ray_weights = sensors.sample_ray_differential(
        time=time,
        sample1=wavelength_sample,
        sample2=pos_unit,
        sample3=aperture_sample
    )
    return rays, ray_weights, pos
