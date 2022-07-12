from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

def step_size_for_marching(step_count, mint, maxt, jittering_enabled):
    if jittering_enabled:
        return (maxt - mint) / step_count
    else:
        return (maxt - mint) / (step_count - 1)

def query_t_for_step(step_j, step_size, jitter_sample, jittering_enabled, mint=0):
    """If jittering is enabled, we use a single jitter sample for the whole ray."""
    if jittering_enabled:
        return mint + step_size * (step_j + jitter_sample)
    else:
        return mint + step_size * step_j


class NeRFIntegrator(mi.ad.integrators.common.RBIntegrator):
    """
    Simplified NeRF-style integrator with emission accumulated along the ray (no scattering).
    Limitations:
    - backed by a dense grid instead of a neural network
    - no direction-dependent emission
    """
    def __init__(self, props=mi.Properties()):
        super().__init__(props=props)

        self.hide_emitters = props.get('hide_emitters', False)

        self.queries_per_ray = props.get('queries_per_ray', 128)
        self.density_noise_std = props.get('density_noise_std', 0.)
        self.jittering_enabled = props.get('jittering_enabled', True)
        self.activation_type = props.get('activation', 'identity').lower()


    def activation(self, raw):
        if self.activation_type in (None, 'identity'):
            return raw
        if self.activation_type == 'relu':
            return dr.maximum(0, raw)

        raise ValueError(f'Unsupported activation: {self.activation_type}')


    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum] = None,
               state_in: Optional[mi.Spectrum] = None,
               active: mi.Bool = None,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:

        primal = (mode == dr.ADMode.Primal)

        ray = mi.Ray3f(ray)
        result = mi.Spectrum(0. if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        throughput = mi.Float(1.)

        # 1. Reach medium bbox
        si = scene.ray_intersect(ray)
        mei = dr.zeros(mi.MediumInteraction3f)
        mei.wavelengths = ray.wavelengths
        mei.medium = si.target_medium(ray.d)

        # We will only simulate rays that encountered a medium
        active = si.is_valid() & dr.neq(mei.medium, None)
        escaped = ~active
        ray[active] = si.spawn_ray(ray.d)

        # 2. Find the other side of the medium bbox
        si[active] = scene.ray_intersect(ray, active)
        active &= si.is_valid()

        # 3. Ray-march inside medium, accumulating emission
        step_size = step_size_for_marching(self.queries_per_ray, 0, si.t,
                                           self.jittering_enabled)
        t_a = mi.Float(0.)
        weights_sum = mi.Float(0.)
        still_walking = mi.Mask(active)
        step_j = mi.UInt32(0)
        jitter_sample = sampler.next_1d(still_walking)

        loop = mi.Loop('NeRF forward raymarching',
                       lambda: (sampler, still_walking, step_j, t_a, result,
                                weights_sum, throughput))
        # "Forward-looking" loop
        # TODO: add linear interpolation
        while loop(still_walking):
            t_b = query_t_for_step(step_j + 1, step_size, jitter_sample, self.jittering_enabled)

            actual_step = t_b - t_a
            with dr.resume_grad(when=not primal):
                sigma, emission = self.query_medium(mei, sampler, ray, t_b, still_walking)

                # Enforce zero density (= fully transparent) at the last traversal step (?)
                # TODO: double-check handling of final step
                alpha_recip = dr.select(step_j + 1 < self.queries_per_ray,
                                        dr.exp(-sigma * actual_step),
                                        1.0)
                weight = (1.0 - alpha_recip) * throughput
                safe_alpha_recip = alpha_recip + 1e-10

            if primal:
                result[still_walking] += weight * emission
            else:
                result[still_walking] -= weight * emission

            step_j[still_walking] = step_j + 1
            t_a[still_walking] = t_b
            still_walking &= step_j < self.queries_per_ray

            throughput[still_walking] *= safe_alpha_recip
            weights_sum[still_walking] += weight

            if mode == dr.ADMode.Backward:
                with dr.resume_grad():
                    dr.backward_from(δL * (
                        # Contribution from emission and sigma_t at this step
                        emission * weight
                        # Contribution from sigma_t at later steps
                        + (result / dr.detach(safe_alpha_recip)) * safe_alpha_recip
                    ))

        # Traverse the other medium boundary.
        # Note: we assume a convex medium boundary, i.e. the medium is
        # encountered at most once per ray.
        ray[active] = si.spawn_ray(ray.d)
        si[active] = scene.ray_intersect(ray, active)

        # 5. Composite with background emitter
        emitter = si.emitter(scene)
        active_e = (escaped | active) & dr.neq(emitter, None)
        if self.hide_emitters:
            # TODO: is that correct pre-multiplied alpha handling?
            active_e &= (weights_sum > 0)

        if True or primal:
            contrib = (1 - weights_sum) * emitter.eval(si, active_e)
            result[active_e] += contrib

        return result, active_e, result


    def query_medium(self, mei_base, sampler, ray, t, active):
        # TODO: avoid this copy
        mei =  mi.MediumInteraction3f(mei_base)
        mei.t = t
        mei.p = ray(t)

        _, _, sigma_t = mei.medium.get_scattering_coefficients(mei, active)
        # Assume non-spectrally varying sigma_t
        sigma_t = self.activation(sigma_t.x)
        if self.density_noise_std > 0:
            assert 'Incorrect for now: noise rnd is wrong on second loop of adjoint'
            sigma_t += mi.warp.square_to_std_normal(sampler.next_2d(active)).x

        emission = mei.medium.get_emission(mei, active)
        return sigma_t, emission


mi.register_integrator("nerf", lambda props: NeRFIntegrator(props))
