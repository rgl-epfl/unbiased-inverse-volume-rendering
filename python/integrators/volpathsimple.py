from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


class VolpathSimpleIntegrator(mi.ad.integrators.common.RBIntegrator):
    """Simplified volumetric path tracer with support for Differential Delta Tracking.
    Some important assumptions are made:
    - There are no surfaces in the scene!
    - There is only one medium in the scene, contained within a convex bounding volume.
    - The only emitter is an infinite light source (e.g. `envmap` or `constant`).
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props=props)

        self.hide_emitters = props.get('hide_emitters', False)
        self.use_drt = props.get('use_drt', True)
        self.use_drt_subsample = props.get('use_drt_subsample', True)
        self.use_drt_mis = props.get('use_drt_mis', True)
        self.use_nee = props.get('use_nee', True)

        # TODO: support Russian Roulette or make it clear that it's disabled

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

        # TODO: support adjoint
        primal = (mode == dr.ADMode.Primal)
        # TODO: support recursive calls with path_state (if really needed)
        path_state = None

        ray = mi.Ray3f(ray)
        wavefront_size = dr.width(ray.d)
        result, throughput = mi.Spectrum(0.0), mi.Spectrum(1.0)
        medium = dr.zeros(mi.MediumPtr, wavefront_size)
        channel = 0

        if path_state is not None:
            # Recursive ray (detached, e.g. for RBP)
            active = path_state.active()
            depth = path_state.depth()
            si = path_state.si()
            assert False, 'Need to handle self.medium'
            medium = dr.select(active, mi.MediumPtr(self.medium), medium)  # Assuming a single medium
            escaped = path_state.escaped()
        else:
            active = mi.Mask(True)
            depth = dr.zeros(mi.Int32, wavefront_size)
            sampler.next_1d(active) # This is the random number normally used to sample a Color channel in C++
            # Intersect medium bbox and traverse the boundary
            si, escaped = self.reach_medium(scene, ray, medium, active)
        # Note: at this point, `active` only includes lanes that do need to traverse the medium.

        if self.use_nee:
            # If `path_state` is given, we kind of assume that `all(active & escaped) == False``,
            # i.e. that a recursive ray is not going to be started for
            # a path that just escaped the bbox. It allows us not to maintain
            # a whole MediumInteraction as part of the recursive state.
            last_scatter_it = dr.zeros(mi.Interaction3f)
            last_scatter_it[escaped] = dr.detach(si)
            if path_state is not None:
                has_scattered = active & (~escaped)
                last_scatter_direction_pdf = path_state.last_scatter_direction_pdf()
            else:
                has_scattered = mi.Mask(False)
                last_scatter_direction_pdf = mi.Float(1.0)
        else:
            has_scattered = None
            last_scatter_it = None
            last_scatter_direction_pdf = None

        loop = mi.Loop("VolpathSimpleSampleLoop", lambda: (
            active, escaped, depth, ray, medium, throughput, si, result, sampler,
            has_scattered, last_scatter_it, last_scatter_direction_pdf))

        while loop(active):
            # Russian Roulette (taking eta = 1.0)
            q = dr.minimum(dr.max(throughput), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= dr.any(dr.neq(throughput, 0.0)) & (
                (~perform_rr | (sampler.next_1d(active) < q)))
            throughput[perform_rr] = throughput * dr.rcp(dr.detach(q))

            # Handle medium sampling and potential medium escape
            needs_differentiable_weight = False #self.use_autodiff
            # For technical reason, it's better to use a single pointer
            # to the only medium in the scene than the `medium` pointer array
            mei, mei_weight = self.sample_real_interaction(
                medium, ray, sampler, channel, active, needs_differentiable_weight)
            throughput[active] = throughput * mei_weight

            did_escape = active & (~mei.is_valid())
            still_in_medium = active & mei.is_valid()

            # Handle null and real scatter events
            did_scatter = mi.Mask(still_in_medium)

            if self.use_nee:
                has_scattered |= did_scatter

            # Real scattering: add effect of albedo
            albedo = medium.get_albedo(mei, did_scatter)
            throughput[did_scatter] = throughput * albedo
            del albedo

            # Rays that have still not escaped but reached max depth
            # are killed inside of the medium (zero contribution)
            depth[did_scatter] = depth + 1
            active &= still_in_medium & (depth < self.max_depth)

            # --- Emitter sampling
            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase = mei.medium.phase_function()
            phase[~did_scatter] = dr.zeros(mi.PhaseFunctionPtr, 1)
            if self.use_nee:
                active_e = did_scatter & active
                nee_contrib = self.sample_emitter_for_nee(
                    mei, scene, sampler, medium, channel, phase_ctx, phase,
                    throughput, active_e)
                result[active_e] = result + dr.detach(nee_contrib)
                del active_e

            # --- Phase function sampling
            # Note: assuming phase_pdf = 1 (perfect importance sampling)
            wo, phase_pdf = phase.sample(
                phase_ctx, mei, sampler.next_1d(did_scatter), sampler.next_2d(did_scatter), did_scatter)
            new_ray = mei.spawn_ray(wo)
            ray[did_scatter] = new_ray

            # Maintain some quantities needed at the end for NEE
            if self.use_nee:
                last_scatter_it[did_scatter] = dr.detach(mei)
                last_scatter_direction_pdf[did_scatter] = dr.detach(phase_pdf)

            # Update ray upper bound (medium boundary on the other side)
            needs_update = did_scatter | did_escape
            si[needs_update] = scene.ray_intersect(ray, needs_update)
            ray.maxt[needs_update] = dr.select(dr.isfinite(si.t), si.t, dr.largest(mi.Float))

            # If a ray was very close to the boundary, it might have
            # accidentally escaped despite sampling an interaction.
            # We just kill the ray as if it was absorbed (won't hit envmap).
            accidental_escape = did_scatter & ~si.is_valid()
            active &= ~accidental_escape

            # --- Handle escaped rays: cross the null boundary
            ray[did_escape] = si.spawn_ray(ray.d)  # Continue on the other side of the boundary
            escaped |= did_escape

        # --- Envmap contribution
        contrib, emitter_pdf, active_e = self.compute_envmap_contribution_and_update_si(
            scene, ray, si, has_scattered, last_scatter_it, depth, escaped)
        if self.use_nee:
            hit_mis_weight = mi.ad.common.mis_weight(last_scatter_direction_pdf, emitter_pdf)
        else:
            hit_mis_weight = 1.0
        result[active_e] = result + throughput * hit_mis_weight * contrib
        del contrib, emitter_pdf, active_e, hit_mis_weight

        return result, active, result



    def reach_medium(self, scene, ray, medium, active):
        """
        In this simplified setting, rays either hit the medium's bbox and
        go in or escape directly to infinity.
        Warning: this function mutates its inputs.
        """
        si = scene.ray_intersect(ray, active)
        escaped = active & (~si.is_valid())
        active &= si.is_valid()
        # By convention, crossing the medium's bbox does *not*
        # count as an interaction (depth++) when it's a null BSDF.

        # Continue on the other side of the boundary and find
        # the opposite side (exit point from the medium).
        ray[active] = si.spawn_ray(ray.d)
        si_new = scene.ray_intersect(ray, active)
        # We might have hit a corner case and escaped despite
        # originally hitting the medium bbox.
        active &= si_new.is_valid()

        medium[active] = si.target_medium(ray.d)
        ray.maxt[active] = dr.select(dr.isfinite(si_new.t), si_new.t, dr.largest(mi.Float))
        si[active] = si_new

        return si, escaped



    def sample_real_interaction(self, medium, ray, sampler, channel, _active,
                                needs_differentiable_weight):
        """
        `Medium::sample_interaction` returns an interaction that could be a null interaction.
        Here, we loop until a real interaction is sampled.

        The given ray's `maxt` value must correspond to the closest surface
        interaction (e.g. medium bounding box) in the direction of the ray.
        """
        # TODO: could make this faster for the homogeneous special case
        # TODO: could make this faster when there's a majorant supergrid
        #       by performing both DDA and "real interaction" sampling in
        #       the same loop.
        # We will keep updating the origin of the ray during traversal.
        running_ray = dr.detach(type(ray)(ray))
        # So we also keep track of the offset w.r.t. the original ray's origin.
        running_t = mi.Float(0.)

        active = mi.Mask(_active)
        weight = mi.Spectrum(1.)
        mei = dr.zeros(mi.MediumInteraction3f, dr.width(ray))
        mei.t = dr.select(active, dr.nan, dr.inf)

        loop = mi.Loop("medium_sample_interaction_real", lambda: (
            active, weight, mei, running_ray, running_t, sampler))
        while loop(active):
            mei_next = medium.sample_interaction(running_ray, sampler.next_1d(active), channel, active)
            if not needs_differentiable_weight:
                mei_next = dr.detach(mei_next)
            mei[active] = mei_next
            mei.t[active] = mei.t + running_t

            majorant = mei_next.combined_extinction[channel]
            r = dr.select(dr.neq(majorant, 0), mei_next.sigma_t[channel] / majorant, 0)

            # Some lanes escaped the medium. Others will continue sampling
            # until they find a real interaction.
            active &= mei_next.is_valid()
            did_null_scatter = active & (sampler.next_1d(active) >= r)

            if needs_differentiable_weight:
                # With DRT and some other variants, this AD-constructed weight is never used.
                # Disabling it enables using symbolic loops.

                # TODO: does this correctly handle surfaces within the medium? (via ray.maxt probably)
                # These lanes found a real interaction
                did_scatter = (~did_null_scatter) & active
                event_pdf = dr.select(dr.neq(mei_next.combined_extinction, 0),
                                    mei_next.sigma_t / mei_next.combined_extinction, 0)
                event_pdf = dr.select(did_scatter, event_pdf, 1. - event_pdf)
                weight[active] = weight * dr.select(
                    dr.neq(event_pdf, 0.),
                    event_pdf / dr.detach(event_pdf),
                    1.
                )

            active &= did_null_scatter
            # Update ray to only sample points further than the
            # current null interaction.
            next_t = dr.detach(mei_next.t)
            running_ray.o[active] = running_ray.o + next_t * running_ray.d
            running_ray.maxt[active] = running_ray.maxt - next_t
            running_t[active] = running_t + next_t


        did_sample = _active & mei.is_valid()
        mei.p = dr.select(did_sample, ray(mei.t), dr.nan)
        mei.mint = mi.Float(dr.nan)  # Value was probably wrong, so we make sure it's unused
        # The final medium property values should be attached,
        # regardless of `needs_differentiable_weight`
        mei.sigma_s, mei.sigma_n, mei.sigma_t = medium.get_scattering_coefficients(mei, did_sample)

        return mei, weight


    def sample_emitter_for_nee(self, mei, scene, sampler, medium, channel,
                               phase_ctx, phase, throughput, active):
        emitted, ds = self.sample_emitter(mei, scene, sampler, medium, channel, active)
        # Evaluate the phase function in the sampled direction.
        # Assume that phase_val == phase_pdf (perfect importance sampling)
        phase_val = phase.eval(phase_ctx, mei, ds.d, active)
        phase_pdf = phase_val

        nee_contrib = throughput * phase_val * mi.ad.common.mis_weight(ds.pdf, phase_pdf) * emitted
        return nee_contrib


    def sample_emitter(self, ref_interaction, scene, sampler, medium, channel, active,
                       adjoint=None):
        """
        Starting from the given `ref_interaction` inside of a medium, samples a direction
        toward an emitter and estimates transmittance with ratio tracking.

        This simplified implementation does not support:
        - presence of surfaces within the medium
        - propagating adjoint radiance (adjoint pass)
        """
        # TODO: also backprop w.r.t. emitter value

        active = mi.Mask(active)
        # Assuming there's a single medium, avoids one vcall
        # medium = self.medium

        dir_sample = sampler.next_2d(active)
        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, dir_sample,
                                                         False, active)
        sampling_worked = dr.neq(ds.pdf, 0.0)
        emitter_val[~sampling_worked] = 0.0
        emitter_val = dr.detach(emitter_val)
        active &= sampling_worked


        # Trace a ray toward the emitter and find the medium's bbox
        # boundary in that direction.
        ray = ref_interaction.spawn_ray(ds.d)
        si = scene.ray_intersect(ray, active)
        ray.maxt = si.t
        transmittance = self.estimate_transmittance(
            ray, 0, si.t, medium, sampler, channel, active & si.is_valid(), adjoint=adjoint)

        return emitter_val * transmittance, ds



    def estimate_transmittance(self, ray_full, tmin, tmax, medium, sampler, channel, active,
                               adjoint=None):
        """Estimate the transmittance between two points along a ray.

        This simplified implementation does not support:
        - presence of surfaces within the medium
        - propagating adjoint radiance (adjoint pass)
        """

        # Support tmax < tmin, but not negative tmin or tmax
        needs_swap = tmax < tmin
        tmp = tmin
        tmin = dr.select(needs_swap, tmax, tmin)
        tmax = dr.select(needs_swap, tmp, tmax)
        del needs_swap, tmp

        active = mi.Mask(active)
        ray = type(ray_full)(ray_full)
        ray.o = ray_full(tmin)
        tmax = tmax - tmin
        ray.maxt = tmax
        del ray_full, tmin

        transmittance = mi.Spectrum(dr.select(active, 1.0, 0.0))

        # --- Estimate transmittance with Ratio Tracking
        # Simplified assuming that we start from within a medium, there's a single
        # medium in the scene and no surfaces.
        loop = mi.Loop("VolpathSimpleNEELoop", lambda: (active, ray, tmax, transmittance, sampler))
        while loop(active):
            # TODO: support majorant supergrid in-line to avoid restarting DDA traversal each time
            # Handle medium interactions / transmittance
            mei = medium.sample_interaction(ray, sampler.next_1d(active), channel, active)
            mei.t[active & (mei.t > tmax)] = dr.inf
            # If interaction falls out of bounds, we don't have anything valid to accumulate
            active &= mei.is_valid()

            # Ratio tracking for transmittance estimation:
            # update throughput estimate with probability of sampling a null-scattering event.
            tr_contribution = dr.select(
                dr.neq(mei.combined_extinction, 0),
                mei.sigma_n / mei.combined_extinction,
                mei.sigma_n)

            if adjoint is not None:
                active_adj = active & (tr_contribution > 0.0)
                dr.backward_from(tr_contribution * dr.select(
                    active_adj, dr.detach(adjoint / tr_contribution), 0.0))

            # Apply effect of this interaction
            transmittance[active] = transmittance * dr.detach(tr_contribution)
            # Adopt newly sampled position in the medium
            ray.o[active] = mei.p
            tmax[active] = tmax - mei.t
            ray.maxt[active] = tmax

            # Continue walking through medium
            active &= dr.any(dr.neq(transmittance, 0.0))

        return transmittance



    def compute_envmap_contribution_and_update_si(self, scene, ray, si, has_scattered,
                                                  last_scatter_it, depth, escaped):
        si_update_needed = escaped & si.is_valid()
        si[si_update_needed] = scene.ray_intersect(ray, si_update_needed)
        # All escaped rays can now query the envmap
        emitter = si.emitter(scene)
        active_e = escaped & dr.neq(emitter, None) & ~((depth <= 0) & self.hide_emitters)

        if self.use_nee:
            assert last_scatter_it is not None
            ds = mi.DirectionSample3f(scene, si, last_scatter_it)
            emitter_pdf = emitter.pdf_direction(last_scatter_it, ds, active_e)
            # MIS should be disabled (i.e. MIS weight = 1) if there wasn't even
            # a valid interaction from which the emitter could have been sampled,
            # e.g. in the case a ray escaped directly.
            emitter_pdf = dr.select(has_scattered, emitter_pdf, 0.0)
        else:
            emitter_pdf = None

        return emitter.eval(si, active_e), emitter_pdf, active_e




mi.register_integrator("volpathsimple", lambda props: VolpathSimpleIntegrator(props))
