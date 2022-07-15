from __future__ import annotations # Delayed parsing of type annotations
import struct

import drjit as dr
import mitsuba as mi

from util import get_single_medium


class VolpathSimpleIntegrator(mi.ad.integrators.common.RBIntegrator):
    """Simplified volumetric path tracer with support for Differential Delta Tracking.
    Some important assumptions are made:
    - There are no surfaces in the scene!
    - There is only one medium in the scene, contained within a convex bounding volume.
    - The medium boundary must use a `null` BSDF
    - The only emitter is an infinite light source (e.g. `envmap` or `constant`).
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props=props)

        self.hide_emitters = props.get('hide_emitters', False)
        # Next event estimation: sample emitters at each volume interaction
        self.use_nee = props.get('use_nee', True)
        # Enable the DRT sampling strategy for in-scattering gradients.
        self.use_drt = props.get('use_drt', True)
        # Use the DRT sampling strategy only once per path.
        # If disabled, the cost of the adjoint will grow quadratically with
        # the path length due to the need to trace a recursive path.
        self.use_drt_subsampling = props.get('use_drt_subsampling', True)
        # In the adjoint, use MIS to combine the specialized transmittance-only
        # sampling technique (DRT) with the standard extinction-weighted transmittance
        # sampling technique.
        self.use_drt_mis = props.get('use_drt_mis', True)

        # TODO: support Russian Roulette or make it clear that it's disabled

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum] = None,
               state_in: Optional[mi.Spectrum] = None,
               active: mi.Bool = None,
               path_state: PathState = None,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:

        primal = (mode == dr.ADMode.Primal)

        ray = mi.Ray3f(ray)
        wavefront_size = dr.width(ray.d)
        result = mi.Spectrum(0. if primal else state_in)
        throughput = mi.Spectrum(1.0)
        medium = get_single_medium(scene)
        channel = 0

        # --- Recursive calls to `sample`: restore state
        if path_state is not None:
            # Recursive ray (detached, e.g. for DRT)
            assert primal, 'Cannot trace attached recursive rays'
            active = path_state.active()
            depth = path_state.depth()
            si = path_state.si()
            escaped = path_state.escaped()
        else:
            active = mi.Mask(True)
            depth = dr.zeros(mi.Int32, wavefront_size)
            sampler.next_1d(active) # This is the random number normally used to sample a Color channel in C++
            # Intersect medium bbox and traverse the boundary
            si, escaped = self.reach_medium(scene, ray, active)
        # Note: at this point, `active` only includes lanes that do need to traverse the medium.

        # --- Prepare next event estimation
        if self.use_nee:
            # If `path_state` is given, we kind of assume that `all(active & escaped) == False``,
            # i.e. that a recursive ray is not going to be started for
            # a path that just escaped the bbox. It allows us not to maintain
            # a whole MediumInteraction as part of the recursive state.
            last_scatter_it = dr.zeros(mi.Interaction3f)
            last_scatter_it[escaped] = si
            if path_state is not None:
                has_scattered = active & (~escaped)
                last_scatter_direction_pdf = path_state.last_scatter_direction_pdf()
            else:
                has_scattered = mi.Mask(False)
                last_scatter_direction_pdf = mi.Float(1.0)
        else:
            has_scattered = last_scatter_it = last_scatter_direction_pdf = None

        # --- Prepare DRT subsampling
        drt_reservoir = None
        if self.use_drt and self.use_drt_subsampling:
            drt_reservoir = DRTReservoir(n=1, active=active)

        alt_sampler = None
        alt_seed_rnd = sampler.next_1d(active)
        if not primal:
            # We need a secondary sampler in order to keep the
            # primary sequence of random numbers identical between
            # the primal and adjoint passes (required by PRB).
            alt_seed = struct.unpack('!I', struct.pack('!f', alt_seed_rnd[0]))[0]
            alt_seed = mi.sample_tea_32(alt_seed, 1)[0]
            alt_sampler = sampler.fork()
            alt_sampler.seed(alt_seed, sampler.wavefront_size())
        del alt_seed_rnd

        loop = mi.Loop("VolpathSimpleSampleLoop", lambda: (
            active, escaped, depth, ray, throughput, si, result, sampler, alt_sampler,
            has_scattered, last_scatter_it, last_scatter_direction_pdf, drt_reservoir))

        while loop(active):

            # Russian Roulette (taking eta = 1.0)
            q = dr.minimum(dr.max(throughput), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= dr.any(dr.neq(throughput, 0.0)) & (
                (~perform_rr | (sampler.next_1d(active) < q)))
            throughput[perform_rr] = throughput * dr.rcp(dr.detach(q))

            # Handle medium sampling and potential medium escape
            # For technical reason, it's better to use a single pointer
            # to the only medium in the scene than the `medium` pointer array
            mei, mei_weight = self.sample_real_interaction(
                medium, ray, sampler, channel, active, primal)
            throughput[active] = throughput * mei_weight

            did_escape = active & (~mei.is_valid())
            still_in_medium = active & mei.is_valid()

            # Handle null and real scatter events
            did_scatter = mi.Mask(still_in_medium)

            if self.use_nee:
                has_scattered |= did_scatter

            # --- Scattering gradients
            with dr.resume_grad(when=not primal):
                albedo = dr.select(did_scatter, medium.get_albedo(mei, did_scatter), 1.0)
            if not primal:
                if self.use_drt:
                    # This time `throughput` does *not* cancel out, since it
                    # will not be included in `drt_Li`.
                    adjoint = (δL * throughput)
                    self.backpropagate_scattering_drt(
                        scene, medium, alt_sampler, ray, si, throughput, depth,
                        channel, adjoint, active, drt_reservoir=drt_reservoir)
                    del adjoint

                if (not self.use_drt) or self.use_drt_mis:
                    drt_mis_weight = 1.0
                    if self.use_drt_mis:
                        # Note: MIS weight hardcoded for the power heuristic
                        s2 = dr.sqr(mei.sigma_t)
                        drt_mis_weight = s2 / (1 + s2)

                    # Without DRT: sampling probability is sigma_t(t) * T(t).
                    # It's exactly this remaining 1/sigma_t factor that DRT
                    # aims to eliminate.
                    inv_pdf = dr.rcp(mei.sigma_t)
                    # Note: a detached `throughput` factor cancels out, since
                    # we need to divide Li by `throughput` as well:
                    #    (δL * throughput) * (sigma_t albedo)
                    #    * (result / (throughput * detach(albedo)))
                    Li = result / dr.maximum(1e-8, albedo)

                    with dr.resume_grad():
                        dr.backward_from(drt_mis_weight * δL * (mei.sigma_t * albedo)
                                         * Li * inv_pdf)
                    del Li, inv_pdf, drt_mis_weight
            # ----------

            # --- Transmittance gradients
            # We resample uniformly along the last step within the medium.
            # Note: this could also be handled by backpropagating through
            # null interactions, but then the 1/sigma_n factor from the
            # pdf also becomes problematic at locations where sigma_t is
            # very close to the majorant.
            if not primal:
                # Note: here, `throughput * albedo` cancelled out:
                #   δL * (throughput * albedo) * (-sigma_t)
                #   * (result / (throughput * albedo))
                adj_weight = δL * result
                self.backpropagate_transmittance(medium, alt_sampler, si, mei, ray,
                                                 adj_weight,
                                                 did_scatter, did_escape)
                del adj_weight
            # ----------

            # Account for albedo on subsequent bounces (no-op if there was no scattering)
            throughput *= albedo
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
                    mei, scene, sampler, medium, channel, phase_ctx, phase, throughput,
                    active_e, primal=primal, δL=δL)
                if primal:
                    result[active_e] += nee_contrib
                else:
                    result[active_e] -= nee_contrib
                del nee_contrib
            # ----------

            # --- Phase function sampling
            # TODO: phase function gradients
            # Note: assuming phase_pdf = 1 (perfect importance sampling)
            wo, phase_pdf = phase.sample(
                phase_ctx, mei, sampler.next_1d(did_scatter), sampler.next_2d(did_scatter), did_scatter)
            new_ray = mei.spawn_ray(wo)
            ray[did_scatter] = new_ray
            # ----------

            # Maintain some quantities needed at the end for MIS with NEE
            if self.use_nee:
                last_scatter_it[did_scatter] = mei
                last_scatter_direction_pdf[did_scatter] = phase_pdf

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
            # ----------

        # --- Finalize DRT subsampling
        if (not primal) and self.use_drt and self.use_drt_subsampling:
            drt_state, subsampling_weight = drt_reservoir.get()
            # Note: the subsampling weight omits a `throughput` term,
            # because it cancels out with the `throughput` factor
            # normally included in `adjoint`.
            # TODO: check this
            adjoint = subsampling_weight * δL
            self.backpropagate_scattering_drt(
                scene, medium, alt_sampler, drt_state.ray(), drt_state.si(),
                None, drt_state.depth(), channel,
                adjoint, drt_state.active(), drt_reservoir=None)


        # --- Envmap contribution
        if primal:
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
                hit_mis_weight = mi.ad.common.mis_weight(last_scatter_direction_pdf, emitter_pdf)
            else:
                emitter_pdf = None
                hit_mis_weight = 1.0

            # TODO: envmap gradients
            contrib = emitter.eval(si, active_e)
            result[active_e] += throughput * hit_mis_weight * contrib

            del si_update_needed, emitter, active_e, contrib


        return result, active, result

    def reach_medium(self, scene, ray, active):
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

        # If we lifted the restriction on the number of media,
        # we would need to get the correct pointers now.
        # medium[active] = si.target_medium(ray.d)

        ray.maxt[active] = dr.select(dr.isfinite(si_new.t), si_new.t, dr.largest(mi.Float))
        si[active] = si_new

        return si, escaped



    def sample_real_interaction(self, medium, ray, sampler, channel, _active, is_primal):
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
            mei_next = medium.sample_interaction(running_ray, sampler.next_1d(active),
                                                 channel, active)
            mei[active] = mei_next
            mei.t[active] = mei.t + running_t

            majorant = mei_next.combined_extinction[channel]
            r = dr.select(dr.neq(majorant, 0), mei_next.sigma_t[channel] / majorant, 0)

            # Some lanes escaped the medium. Others will continue sampling
            # until they find a real interaction.
            active &= mei_next.is_valid()
            did_null_scatter = active & (sampler.next_1d(active) >= r)

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
        with dr.resume_grad(when=not is_primal):
            mei.sigma_s, mei.sigma_n, mei.sigma_t = \
                medium.get_scattering_coefficients(mei, did_sample)

        return mei, weight


    def sample_emitter_for_nee(self, mei, scene, sampler, medium, channel,
                               phase_ctx, phase, throughput, active, primal=True, δL=None):
        if not primal:
            nee_sampler = sampler.clone()

        emitted, ds = self.sample_emitter(mei, scene, sampler, medium, channel, active)
        # Evaluate the phase function in the sampled direction.
        # Assume that phase_val == phase_pdf (perfect importance sampling)
        phase_val = phase.eval(phase_ctx, mei, ds.d, active)
        phase_pdf = phase_val

        nee_contrib = throughput * phase_val * mi.ad.common.mis_weight(ds.pdf, phase_pdf) * emitted

        if not primal:
            # Transmittance gradients due to the emitter sample's attenuation
            # by the medium. We re-run transmittance estimation here as another
            # application of path replay backpropagation.
            # TODO: any chance to avoid the second run?
            # TODO: should the adjoint include the phase_val, etc?
            adjoint = δL * nee_contrib
            self.sample_emitter(mei, scene, nee_sampler, medium, channel, active,
                                adjoint=adjoint)

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
        active = mi.Mask(active)

        dir_sample = sampler.next_2d(active)
        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, dir_sample,
                                                         False, active)
        sampling_worked = dr.neq(ds.pdf, 0.0)
        emitter_val &= sampling_worked
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
            with dr.resume_grad(when=adjoint is not None):
                mei = medium.sample_interaction(
                    ray, sampler.next_1d(active), channel, active)
                # Ratio tracking for transmittance estimation:
                # update throughput estimate with probability of sampling a null-scattering event.
                tr_contribution = dr.select(
                    dr.neq(mei.combined_extinction, 0),
                    mei.sigma_n / mei.combined_extinction,
                    mei.sigma_n)

            # If interaction falls out of bounds, we don't have anything
            # valid to accumulate.
            mei.t[active & (mei.t > tmax)] = dr.inf
            active &= mei.is_valid()

            if adjoint is not None:
                # Here again, a `transmittance` factor cancels out:
                #   (δL * transmittance) * contrib
                #   * (Li / (transmittance * detach(contrib)))
                active_adj = active & (tr_contribution > 0.0)
                with dr.resume_grad():
                    dr.backward_from(dr.select(
                        active_adj,
                        adjoint * tr_contribution / dr.detach(tr_contribution),
                        0.0))

            # Apply effect of this interaction
            transmittance[active] *= tr_contribution
            # Adopt newly sampled position in the medium
            ray.o[active] = mei.p
            tmax[active] = tmax - mei.t
            ray.maxt[active] = tmax

            # Continue walking through medium
            active &= dr.any(dr.neq(transmittance, 0.0))

        return transmittance


    def backpropagate_scattering_drt(self, scene, medium, alt_sampler, ray, si,
                                     throughput, depth, channel,
                                     adjoint, active, drt_reservoir=None):
        """
        Estimate in-scattering gradients with Differential Delta Tracking.
        """

        # --- DRT subsampling: at each bounce, instead of applying
        # differential ratio tracking, simply update a reservoir to
        # pick *one* path depth (can be different for each lane) at
        # which to trigger DRT. This function will be called again
        # once all bounces have been simulated, # this time with
        # arguments taken from the DRT reservoir.
        if drt_reservoir is not None:
            assert self.use_drt_subsampling
            state_for_delayed = {
                'depth': depth,
                'si': si,
                'escaped': mi.Mask(False),
                'active': active,
                'ray': ray,
                # This won't be used, we'll have a PDF when picking
                # our own recursive ray direction later.
                'last_scatter_direction_pdf': dr.zeros(mi.Float),
            }

            # TODO: consider using the MIS weight as part of the depth sampling weight
            drt_reservoir.update(state=state_for_delayed, weight=throughput,
                                 sample=alt_sampler.next_1d(active), active=active)
            # No backpropagation at this point, we'll trace all recursive rays
            # at once when all paths have completed.
            return
        # ----------

        # TODO: check if this is needed or we can use the main ray
        sub_ray = mi.Ray3f(ray)
        sub_ray.maxt[active] = dr.select(
            dr.isfinite(si.t), si.t, dr.largest(mi.Float))
        del ray


        # With DRT, the sampling probability is T(t').
        mei_sub, drt_weight = medium.sample_interaction_drt(
            sub_ray, alt_sampler, channel, active)
        with dr.resume_grad():
            mei_sub.sigma_s, mei_sub.sigma_n, mei_sub.sigma_t = \
                medium.get_scattering_coefficients(mei_sub, active);
            mei_sub.combined_extinction = medium.get_majorant(mei_sub, active);

        # This should always succeed.
        active = active & mei_sub.is_valid()

        # --- Estimate incident radiance
        # Unfortunately, this is a new free-flight distance t',
        # different from the main path. Therefore the value Li
        # provided by path replay is not valid. We have to
        # estimate Li again, which is costly.
        with dr.suspend_grad():
            drt_Li = self.sample_recursive(
                scene, alt_sampler, medium, sub_ray, si,
                mei_sub, channel, depth, active)
        # ----------

        if self.use_drt_mis:
            # Note: MIS weight hardcoded for the power heuristic
            drt_mis_weight = 1 / (1 + dr.sqr(mei_sub.sigma_t))
        else:
            drt_mis_weight = 1.0

        with dr.resume_grad():
            albedo_sub = medium.get_albedo(mei_sub, active)
            to_backward = dr.select(active, mei_sub.sigma_t * albedo_sub, 0.)
            dr.backward_from(drt_mis_weight * drt_weight *
                             adjoint * to_backward * drt_Li)


    def backpropagate_transmittance(self, medium, alt_sampler, si, mei, ray,
                                    adj_weight,
                                    did_scatter, did_escape, n_samples=4):
        active = did_scatter | did_escape
        interval = dr.select(did_escape, si.t, mei.t)

        mei_sub = mi.MediumInteraction3f(mei)
        contribs = mi.Spectrum(0.)
        # Pick `n_samples` uniformly over the interval
        # TODO: consider stratified sampling
        for _ in range(n_samples):
            mei_sub.t = alt_sampler.next_1d(active) * interval
            mei_sub.p = ray(mei_sub.t)
            with dr.resume_grad():
                _, _, sigma_t_sub = medium.get_scattering_coefficients(mei_sub, active)
                # The higher sigma_t, the lower the transmittance
                contribs -= sigma_t_sub
        del mei_sub

        # Probability of sampling each of the new distances
        inv_pdf = interval / n_samples
        with dr.resume_grad():
            contribs = dr.select(active, contribs, 0.)
            dr.backward_from(adj_weight * contribs * inv_pdf)


    def sample_recursive(self, scene, alt_sampler, medium, ray, si, mei, channel,
                         depth, active):
        """
        Trace a detached recursive ray to estimate Li incident to the current medium
        interaction.
        """
        result = dr.zeros(mi.Spectrum)
        phase_ctx = mi.PhaseFunctionContext(alt_sampler)
        phase = medium.phase_function()

        # 1. Emitter sampling (including the MIS weight)
        if self.use_nee:
            result += self.sample_emitter_for_nee(
                mei, scene, alt_sampler, medium, channel, phase_ctx,
                phase, mi.Spectrum(1), active)

        # 2. Phase sampling (including the MIS weight)
        # Prepare recursive ray to be traced. We can assume it's leaving
        # from a valid medium interaction.
        # Note: we assume perfect importance sampling of the phase function.
        wo, phase_pdf = phase.sample(
            phase_ctx, mei,
            alt_sampler.next_1d(active), alt_sampler.next_2d(active), active)
        rec_ray = mei.spawn_ray(wo)
        rec_ray = dr.select(active, rec_ray, ray)

        # -- Case 2: ray escaped the medium and must cross the medium boundary
        si_next = scene.ray_intersect(rec_ray, active)
        # Important for homogeneous media
        rec_ray.maxt[active] = dr.select(dr.isfinite(si_next.t),
                                         si_next.t, dr.largest(mi.Float))
        next_depth = dr.select(active, depth + 1, depth)
        path_state = PathState(
            depth=next_depth,
            si=si_next,
            last_scatter_direction_pdf=dr.select(active, phase_pdf, 1.0),
            escaped=mi.Mask(False),
            active=active & (next_depth < self.max_depth),
        )
        # ----------

        Li, _, _ = self.sample(dr.ADMode.Primal, scene, alt_sampler, rec_ray, active=active,
                               path_state=path_state)
        result += Li

        return result & active




class PathState():
    """
    Helper structure holding path state information needed to
    trace recursive rays.
    """
    def __init__(self, depth=None, si=None, last_scatter_direction_pdf=None,
                 escaped=None, active=None, n=None):
        # TODO: would need medium pointer as well if we didn't assume a single medium
        if n is not None:
            assert (depth, si, last_scatter_direction_pdf, escaped) == (None, None, None, None)
            assert active is not None
            self._depth = dr.full(mi.Int32, -1, n)
            self._si = dr.zeros(mi.SurfaceInteraction3f, n)
            self._last_scatter_direction_pdf = dr.zeros(mi.Float, n)
            self._escaped = dr.empty(mi.Mask, n)
            self._active = mi.Mask(active)
        else:
            self._depth = depth
            self._si = si
            self._last_scatter_direction_pdf = last_scatter_direction_pdf
            self._escaped = escaped
            self._active = active

    def loop_put(self, loop):
        loop.put(lambda: (self._depth, self._si, self._last_scatter_direction_pdf,
                          self._escaped, self._active))

    def set(self, state: dict, enabled):
        self._depth[enabled] = state['depth']
        self._si[enabled] = state['si']
        self._last_scatter_direction_pdf[enabled] = state['last_scatter_direction_pdf']
        self._escaped[enabled] = state['escaped']
        self._active[enabled] = state['active']

    def is_valid(self):
        return dr.neq(self._depth, -1)

    def depth(self):
        return mi.Int32(self._depth)
    def si(self):
        return mi.SurfaceInteraction3f(self._si)
    def last_scatter_direction_pdf(self):
        return mi.Float(self._last_scatter_direction_pdf)
    def active(self):
        return mi.Mask(self._active)
    def escaped(self):
        return mi.Mask(self._escaped)



class DRTPathState(PathState):
    def __init__(self, n=None, ray=None, **kwargs):
        super().__init__(n=n, **kwargs)
        if ray is None:
            self._ray = dr.zeros(mi.Ray3f, n)
        else:
            self._ray = ray

    def loop_put(self, loop):
        super().loop_put(loop)
        loop.put(lambda: (self._ray,))

    def set(self, state: dict, enabled):
        super().set(state, enabled)
        self._ray[enabled] = state['ray']

    def ray(self):
        return type(self._ray)(self._ray)


class DRTReservoir():
    """
    Helper class to sample one (or more) depth values along a path
    with Reservoir sampling.
    """
    def __init__(self, n, active):
        assert n == 1, 'Not supported yet: reservoir with size > 1'

        self.n = n
        self.state = DRTPathState(n=n, active=active)
        # Sum of weights seen
        self.wsum = dr.zeros(mi.Spectrum, dr.width(active))
        # Weight of the sample currently in the reservoir
        self.current_weight = dr.zeros(mi.Spectrum, dr.width(active))

    def update(self, state: dict, weight, sample, active):
        assert isinstance(weight, (mi.Spectrum, dr.detached_t(mi.Spectrum))), type(weight)

        weight = dr.select(active, dr.detach(weight), 0)
        self.wsum[active] = self.wsum + weight

        change = active & (sample <= dr.mean(weight / self.wsum))
        self.current_weight[change] = weight
        self.state.set(state, change)


    def get(self):
        d = dr.mean(self.current_weight)
        sampling_weight = dr.select(
            dr.neq(d, 0), dr.mean(self.wsum) * self.current_weight / d, 0)
        return self.state, sampling_weight


    def loop_put(self, loop):
        self.state.loop_put(loop)
        loop.put(lambda: (self.wsum, self.current_weight))



mi.register_integrator("volpathsimple", lambda props: VolpathSimpleIntegrator(props))
