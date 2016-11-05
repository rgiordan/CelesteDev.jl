#module PSFConvolution

using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.get_expected_pixel_brightness!
using Celeste.Model.populate_fsm_vecs!
using Celeste.Model.populate_gal_fsm!
using Celeste.Model.populate_fsm!
using Celeste.Model.ParamSet
using SensitiveFloats.SensitiveFloat
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent

using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!

using DeterministicVI.add_sources_sf!
using DeterministicVI.add_elbo_log_term!
using DeterministicVI.add_scaled_sfs!
using DeterministicVI.ElboArgs
using DeterministicVI.ActivePixel
using DeterministicVI.ImageTile
using DeterministicVI.SourceBrightness
using DeterministicVI.StarPosParams
using DeterministicVI.GalaxyPosParams
using DeterministicVI.CanonicalParams
using DeterministicVI.accumulate_source_pixel_brightness!

using StaticArrays

typealias GMatrix Matrix{SensitiveFloat{CanonicalParams, Float64}}
typealias fs0mMatrix Matrix{SensitiveFloat{StarPosParams, Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{GalaxyPosParams, Float64}}


# TODO: copy over
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
# using PSFConvolution.FSMSensitiveFloatMatrices
# using PSFConvolution.initialize_fsm_sf_matrices!

using Celeste.DeterministicVI.ElboArgs
using Celeste.DeterministicVI.ElboIntermediateVariables
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using Celeste.Model.populate_fsm!
# using PSFConvolution.FSMSensitiveFloatMatrices
# using PSFConvolution.accumulate_band_in_elbo!


type FSMSensitiveFloatMatrices
    # The lower corner of the image (in terms of index values)
    h_lower::Int
    w_lower::Int

    fs1m_image::fs1mMatrix;
    fs1m_image_padded::fs1mMatrix;
    fs1m_conv::fs1mMatrix;
    fs1m_conv_padded::fs1mMatrix;

    # We convolve the star fs0m directly using Lanczos interpolation.
    fs0m_conv::fs0mMatrix;

    E_G::GMatrix
    var_G::GMatrix

    # A vector of psfs, one for each source.
    psf_fft_vec::Vector{Matrix{Complex{Float64}}}
    psf_vec::Vector{Matrix{Float64}}

    # The amount of padding introduced by the convolution on one side of the
    # image (the total pixels added in each dimension are twice this)
    pad_pix_h::Int
    pad_pix_w::Int

    FSMSensitiveFloatMatrices() = begin
        new(1, 1,
            fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs1mMatrix(), fs0mMatrix(),
            GMatrix(), GMatrix(),
            Vector{Matrix{Complex{Float64}}}(),
            Vector{Matrix{Float64}}(),
            0, 0)
    end
end


function initialize_fsm_sf_matrices!(
    fsm_vec::Vector{FSMSensitiveFloatMatrices},
    ea::ElboArgs{Float64},
    psf_image_mat::Matrix{Matrix{Float64}})

    # Get the extreme active pixels in each band.
    h_lower = Int[typemax(Int) for b in ea.images ]
    w_lower = Int[typemax(Int) for b in ea.images ]
    h_upper = Int[0 for b in ea.images ]
    w_upper = Int[0 for b in ea.images ]

    # TODO: no need to correspond to tile boundaries
    for pixel in ea.active_pixels
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        h_lower[pixel.n] = min(h_lower[pixel.n], tile.h_range.start)
        h_upper[pixel.n] = max(h_upper[pixel.n], tile.h_range.stop)
        w_lower[pixel.n] = min(w_lower[pixel.n], tile.w_range.start)
        w_upper[pixel.n] = max(w_upper[pixel.n], tile.w_range.stop)
    end

    sa_n = length(ea.active_sources)

    # Pre-allocate arrays.
    for b in 1:ea.N
        # Require that every PSF is the same size, since we are only
        # keeping one padded matrix for the whole band.
        psf_sizes = Set([ size(im) for im in psf_image_mat[:, b] ])
        @assert length(psf_sizes) == 1
        psf_size = pop!(psf_sizes)

        fsm_vec[b].h_lower = h_lower[b]
        fsm_vec[b].w_lower = w_lower[b]

        h_width = h_upper[b] - h_lower[b] + 1
        w_width = w_upper[b] - w_lower[b] + 1

        # An fsm value is only sensitive to one source's parameters.
        fsm_vec[b].fs1m_image = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, h_width, w_width);
        fsm_vec[b].fs1m_conv = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, h_width, w_width);

        fsm_vec[b].fs0m_conv = zero_sensitive_float_array(
            StarPosParams, Float64, 1, h_width, w_width);

        # The amount of padding introduced by the convolution.  Make sure
        # that the PSF has an odd dimension.
        (fft_size1, fft_size2) =
            (h_width + psf_size[1] - 1, w_width + psf_size[2] - 1)
        @assert psf_size[1] % 2 == 1
        @assert psf_size[2] % 2 == 1
        fsm_vec[b].pad_pix_h = Integer((psf_size[1] - 1) / 2)
        fsm_vec[b].pad_pix_w = Integer((psf_size[2] - 1) / 2)

        fsm_vec[b].fs1m_image_padded = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, fft_size1, fft_size2);
        fsm_vec[b].fs1m_conv_padded = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, fft_size1, fft_size2);

        # Brightness images
        fsm_vec[b].E_G = zero_sensitive_float_array(
            CanonicalParams, Float64, sa_n, h_width, w_width);
        fsm_vec[b].var_G = zero_sensitive_float_array(
            CanonicalParams, Float64, sa_n, h_width, w_width);

        # Store the psf image and its FFT.
        S = size(psf_image_mat, 1)
        fsm_vec[b].psf_fft_vec = Array(Matrix{Complex{Float64}}, S)
        fsm_vec[b].psf_vec = Array(Matrix{Float64}, S)
        for s in 1:size(psf_image_mat, 1)
            fsm_vec[b].psf_fft_vec[s] =
                zeros(Complex{Float64}, fft_size1, fft_size2);
            fsm_vec[b].psf_fft_vec[s][1:psf_size[1], 1:psf_size[2]] =
                psf_image_mat[s, b];
            fft!(fsm_vec[b].psf_fft_vec[s]);
            fsm_vec[b].psf_vec[s] = psf_image_mat[s, b]
        end
    end
end


########################################
# New versions of exisitng functions

# using PSFConvolution.populate_fsm_image!
# using PSFConvolution.populate_source_band_brightness!
# This is just for debugging.
function populate_fsm_vec!(
    ea::ElboArgs, elbo_vars::ElboIntermediateVariables,
    fsm_vec::Array{FSMSensitiveFloatMatrices},
    lanczos_width::Int)

    sbs = load_source_brightnesses(ea,
        calculate_derivs=elbo_vars.calculate_derivs,
        calculate_hessian=elbo_vars.calculate_hessian);

    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        gal_mcs_vec[b] = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian);
    end

    for s in 1:ea.S
    end

    for b=1:ea.N
        for s in 1:ea.S
            populate_star_fsm_image!(
                ea, elbo_vars, s, b, fsm_vec[b].psf_vec[s], fsm_vec[b].fs0m_conv,
                fsm_vec[b].h_lower, fsm_vec[b].w_lower, lanczos_width)
            populate_gal_fsm_image!(ea, elbo_vars, s, b, gal_mcs_vec[b], fsm_vec[b])
            populate_source_band_brightness!(ea, elbo_vars, s, b, fsm_vec[b], sbs[s])
        end
    end
end


# just a version of load_active_pixels! that doesn't discard NaN
using DeterministicVI.ElboArgs
function load_active_pixels!(ea::ElboArgs{Float64},
                             discard_nan::Bool;
                             noise_fraction=0.1,
                             min_radius_pix=8.0)
    @assert length(ea.active_sources) == 1
    s = ea.active_sources[1]

    @assert(length(ea.active_pixels) == 0)

    for n = 1:ea.N
        tiles = ea.images[n].tiles

        patch = ea.patches[s, n]
        pix_loc = Model.linear_world_to_pix(patch.wcs_jacobian,
                                            patch.center,
                                            patch.pixel_center,
                                            ea.vp[s][ids.u])

        # TODO: just loop over the tiles/pixels near the active source
        for t in 1:length(tiles)
            tile = tiles[t]

            tile_source_map = ea.tile_source_map[n][t]
            if s in tile_source_map
                # TODO; use log_prob.jl in the Model module to get the
                # get the expected brightness, not variational inference
                pred_tile_pixels =
                    DeterministicVI.tile_predicted_image(tile, ea, [ s ],
                                                   include_epsilon=false)
                for h in tile.h_range, w in tile.w_range
                    pixel_nan = discard_nan && isnan(tile.pixels[h_im, w_im])
                    if !pixel_nan
                        # The pixel location in the rendered image.
                        h_im = h - minimum(tile.h_range) + 1
                        w_im = w - minimum(tile.w_range) + 1

                        bright_pixel = pred_tile_pixels[h_im, w_im] >
                           tile.iota_vec[h_im] * tile.epsilon_mat[h_im, w_im] * noise_fraction
                        close_pixel =
                            (h - pix_loc[1]) ^ 2 + (w - pix_loc[2])^2 < min_radius_pix^2
                        if bright_pixel || close_pixel
                            push!(ea.active_pixels, ActivePixel(n, t, h_im, w_im))
                        end
                    end
                end
            end
        end
    end
end

###################

include("/home/rgiordan/Documents/git_repos/CelesteDev.jl/rasterized_psf/sensitive_float_fft.jl")


"""
Convolve a populated set of SensitiveFloat matrices in fsms with the PSF
and store them in the matching fs*m_conv SensitiveFloat matrices.
"""
function convolve_fs1m_image!(fsms::FSMSensitiveFloatMatrices, s::Int)
    for h in 1:size(fsms.fs1m_image, 1), w in 1:size(fsms.fs1m_image, 2)
        fsms.fs1m_image_padded[h, w] = fsms.fs1m_image[h, w];
    end

    convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft_vec[s], fsms.fs1m_conv_padded);

    for h in 1:size(fsms.fs1m_image, 1), w in 1:size(fsms.fs1m_image, 2)
        fsms.fs1m_conv[h, w] =
            fsms.fs1m_conv_padded[fsms.pad_pix_h + h, fsms.pad_pix_w + w];
    end

    # Set return type
    return true
end


function clear_fs1m!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.fs1m_image clear!(sf) end
    for sf in fsms.fs1m_image_padded clear!(sf) end
    for sf in fsms.fs1m_conv clear!(sf) end
    for sf in fsms.fs1m_conv_padded clear!(sf) end
end


function clear_brightness!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.E_G clear!(sf) end
    for sf in fsms.var_G clear!(sf) end
end


"""
Populate the fs1m shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_gal_fsm_image!(
            ea::ElboArgs{Float64},
            elbo_vars::ElboIntermediateVariables{Float64},
            s::Int,
            b::Int,
            gal_mcs::Array{GalaxyCacheComponent{Float64}, 4},
            fsms::FSMSensitiveFloatMatrices)
    clear_fs1m!(fsms)
    for pixel in ea.active_pixels
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        if pixel.n == b && s in tile_sources
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            h_fsm = tile.h_range[pixel.h] - fsms.h_lower + 1
            w_fsm = tile.w_range[pixel.w] - fsms.w_lower + 1

            x = SVector{2, Float64}([tile.h_range[pixel.h], tile.w_range[pixel.w]])
            populate_gal_fsm!(elbo_vars.bvn_derivs,
                              fsms.fs1m_image[h_fsm, w_fsm],
                              true, true,
                              s, x, true, Inf,
                              ea.patches[s, b].wcs_jacobian,
                              gal_mcs)
        end
    end
    convolve_fs1m_image!(fsms, s);
end


"""
Populate the fs1m shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_star_fsm_image!(
            ea::ElboArgs{Float64},
            elbo_vars::ElboIntermediateVariables{Float64},
            s::Int,
            b::Int,
            psf_image::Matrix{Float64},
            fs0m_conv::fs0mMatrix,
            h_lower::Int, w_lower::Int,
            lanczos_width::Int)
    for sf in fs0m_conv clear!(sf) end
    # The pixel location of the star.
    star_loc_pix =
        Model.linear_world_to_pix(ea.patches[s, b].wcs_jacobian,
                                  ea.patches[s, b].center,
                                  ea.patches[s, b].pixel_center,
                                  ea.vp[s][lidx.u]) -
        Float64[ h_lower - 1, w_lower - 1]
    lanczos_interpolate!(fs0m_conv, psf_image, star_loc_pix, lanczos_width,
                         ea.patches[s, b].wcs_jacobian, true);
end


"""
Updates fsms.E_G and fsms.var_G in place with the contributions from this
source in this band.
"""
function populate_source_band_brightness!(
    ea::ElboArgs{Float64},
    elbo_vars::ElboIntermediateVariables{Float64},
    s::Int,
    b::Int,
    fsms::FSMSensitiveFloatMatrices,
    sb::SourceBrightness{Float64})

    # For now s is fixed, and only doing one band.
    # for s in tile_sources
    active_source = s in ea.active_sources
    calculate_hessian =
        elbo_vars.calculate_hessian && elbo_vars.calculate_derivs &&
        active_source

    for pixel in ea.active_pixels
        if pixel.n == b
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            this_pixel = tile.pixels[pixel.h, pixel.w]

            # These are indices within the fs?m image.
            h_fsm = tile.h_range[pixel.h] - fsms.h_lower + 1
            w_fsm = tile.w_range[pixel.w] - fsms.w_lower + 1

            accumulate_source_pixel_brightness!(
                                elbo_vars, ea,
                                fsms.E_G[h_fsm, w_fsm],
                                fsms.var_G[h_fsm, w_fsm],
                                fsms.fs0m_conv[h_fsm, w_fsm],
                                fsms.fs1m_conv[h_fsm, w_fsm],
                                sb, b, s, active_source)
        end
    end
end


"""
Uses the values in fsms to add the contribution from this band to the ELBO.
"""
function accumulate_band_in_elbo!(
    ea::ElboArgs{Float64},
    elbo_vars::ElboIntermediateVariables{Float64},
    fsms::FSMSensitiveFloatMatrices,
    sbs::Vector{SourceBrightness{Float64}},
    gal_mcs_vec::Array{Array{GalaxyCacheComponent{Float64}, 4}},
    b::Int, lanczos_width::Int)

    clear_brightness!(fsms)

    for s in 1:ea.S
        populate_star_fsm_image!(
            ea, elbo_vars, s, b, fsms.psf_vec[s], fsms.fs0m_conv,
            fsms.h_lower, fsms.w_lower, lanczos_width)
        populate_gal_fsm_image!(ea, elbo_vars, s, b, gal_mcs_vec[b], fsms)
        populate_source_band_brightness!(ea, elbo_vars, s, b, fsms, sbs[s])
    end

    for pixel in ea.active_pixels
        if pixel.n == b
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            this_pixel = tile.pixels[pixel.h, pixel.w]
            if !Base.isnan(this_pixel)
                # These are indices within the fs?m image.
                h_fsm = tile.h_range[pixel.h] - fsms.h_lower + 1
                w_fsm = tile.w_range[pixel.w] - fsms.w_lower + 1

                E_G = fsms.E_G[h_fsm, w_fsm]
                var_G = fsms.var_G[h_fsm, w_fsm]

                # There are no derivatives with respect to epsilon, so can safely add
                # to the value.
                E_G.v[1] += tile.epsilon_mat[pixel.h, pixel.w]

                # Add the terms to the elbo given the brightness.
                iota = tile.iota_vec[pixel.h]
                println((b, h_fsm, w_fsm,
                         E_G.v[1],
                         fsms.fs0m_conv[h_fsm, w_fsm].v[1],
                         fsms.fs1m_conv[h_fsm, w_fsm].v[1],
                         tile.epsilon_mat[pixel.h, pixel.w]))
                add_elbo_log_term!(elbo_vars, E_G, var_G, elbo_vars.elbo, this_pixel, iota)
                add_scaled_sfs!(elbo_vars.elbo, E_G, -iota,
                                elbo_vars.calculate_hessian &&
                                elbo_vars.calculate_derivs)

                # Subtract the log factorial term. This is not a function of the
                # parameters so the derivatives don't need to be updated. Note that
                # even though this does not affect the ELBO's maximum, it affects
                # the optimization convergence criterion, so I will leave it in for now.
                elbo_vars.elbo.v[1] -= lfact(this_pixel)
            end
        end
    end
end


function elbo_likelihood_with_fft!(
    ea::ElboArgs, elbo_vars::ElboIntermediateVariables,
    lanczos_width::Int64,
    fsm_vec::Array{FSMSensitiveFloatMatrices})

    sbs = load_source_brightnesses(ea,
        calculate_derivs=elbo_vars.calculate_derivs,
        calculate_hessian=elbo_vars.calculate_hessian);

    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        gal_mcs_vec[b] = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian);
    end

    clear!(elbo_vars.elbo)
    for b in 1:ea.N
        accumulate_band_in_elbo!(ea, elbo_vars, fsm_vec[b], sbs, gal_mcs_vec,
                                 b, lanczos_width)
    end
end

using StaticArrays


#end
