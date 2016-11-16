module ELBOPixelatedPSF

using DeterministicVI.ElboArgs
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.SourceBrightness
using DeterministicVI.StarPosParams
using DeterministicVI.GalaxyPosParams
using DeterministicVI.CanonicalParams
using DeterministicVI.GalaxyComponent
using DeterministicVI.SkyPatch

using Celeste.Model.populate_gal_fsm!
using Celeste.Model.ParamSet
using Celeste.Model.linear_world_to_pix

using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!

using DeterministicVI.load_source_brightnesses
using DeterministicVI.add_sources_sf!
using DeterministicVI.add_elbo_log_term!
using DeterministicVI.add_scaled_sfs!
using DeterministicVI.accumulate_source_pixel_brightness!

using StaticArrays

typealias GMatrix Matrix{SensitiveFloat{CanonicalParams, Float64}}
typealias fs0mMatrix Matrix{SensitiveFloat{StarPosParams, Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{GalaxyPosParams, Float64}}


const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"
include(joinpath(dir, "rasterized_psf/sensitive_float_fft.jl"))
include(joinpath(dir, "rasterized_psf/lanczos.jl"))

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


function initialize_fsm_sf_matrices_band!(
    fsms::FSMSensitiveFloatMatrices,
    b::Int,
    num_active_sources::Int,
    h_lower::Int, w_lower::Int,
    h_upper::Int, w_upper::Int,
    psf_image_mat::Matrix{Matrix{Float64}})

    # Require that every PSF is the same size, since we are only
    # keeping one padded matrix for the whole band.
    psf_sizes = Set([ size(im) for im in psf_image_mat[:, b] ])
    @assert length(psf_sizes) == 1
    psf_size = pop!(psf_sizes)

    fsms.h_lower = h_lower[b]
    fsms.w_lower = w_lower[b]

    h_width = h_upper[b] - h_lower[b] + 1
    w_width = w_upper[b] - w_lower[b] + 1

    # An fsm value is only sensitive to one source's parameters.
    fsms.fs1m_image = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, h_width, w_width);
    fsms.fs1m_conv = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, h_width, w_width);

    fsms.fs0m_conv = zero_sensitive_float_array(
        StarPosParams, Float64, 1, h_width, w_width);

    # The amount of padding introduced by the convolution
    (fft_size1, fft_size2) =
        (h_width + psf_size[1] - 1, w_width + psf_size[2] - 1)
    # Make sure that the PSF has an odd dimension.
    @assert psf_size[1] % 2 == 1
    @assert psf_size[2] % 2 == 1
    fsms.pad_pix_h = Integer((psf_size[1] - 1) / 2)
    fsms.pad_pix_w = Integer((psf_size[2] - 1) / 2)

    fsms.fs1m_image_padded = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, fft_size1, fft_size2);
    fsms.fs1m_conv_padded = zero_sensitive_float_array(
        GalaxyPosParams, Float64, 1, fft_size1, fft_size2);

    # Brightness images
    fsms.E_G = zero_sensitive_float_array(
        CanonicalParams, Float64, num_active_sources, h_width, w_width);
    fsms.var_G = zero_sensitive_float_array(
        CanonicalParams, Float64, num_active_sources, h_width, w_width);

    # Store the psf image and its FFT.
    S = size(psf_image_mat, 1)
    fsms.psf_fft_vec = Array(Matrix{Complex{Float64}}, S)
    fsms.psf_vec = Array(Matrix{Float64}, S)
    for s in 1:size(psf_image_mat, 1)
        fsms.psf_fft_vec[s] =
            zeros(Complex{Float64}, fft_size1, fft_size2);
        fsms.psf_fft_vec[s][1:psf_size[1], 1:psf_size[2]] =
            psf_image_mat[s, b];
        fft!(fsms.psf_fft_vec[s]);
        fsms.psf_vec[s] = psf_image_mat[s, b]
    end
end


function initialize_fsm_sf_matrices!(
    fsm_vec::Vector{FSMSensitiveFloatMatrices},
    ea::ElboArgs{Float64},
    psf_image_mat::Matrix{Matrix{Float64}})

    # Get the extreme active pixels in each band.
    h_lower_vec = Int[typemax(Int) for b in ea.images ]
    w_lower_vec = Int[typemax(Int) for b in ea.images ]
    h_upper_vec = Int[0 for b in ea.images ]
    w_upper_vec = Int[0 for b in ea.images ]

    # TODO: no need to correspond to tile boundaries
    for pixel in ea.active_pixels
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        h_lower_vec[pixel.n] = min(h_lower_vec[pixel.n], tile.h_range.start)
        h_upper_vec[pixel.n] = max(h_upper_vec[pixel.n], tile.h_range.stop)
        w_lower_vec[pixel.n] = min(w_lower_vec[pixel.n], tile.w_range.start)
        w_upper_vec[pixel.n] = max(w_upper_vec[pixel.n], tile.w_range.stop)
    end

    num_active_sources = length(ea.active_sources)

    # Pre-allocate arrays.
    for b in 1:ea.N
        initialize_fsm_sf_matrices_band!(
            fsm_vec[b], b, num_active_sources,
            h_lower_vec[b], w_lower_vec[b],
            h_upper_vec[b], w_upper_vec[b],
            psf_image_mat)
    end
end


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
            populate_gal_fsm_image!(
                ea, elbo_vars, s, b, gal_mcs_vec[b], fsm_vec[b])
            populate_source_band_brightness!(
                ea, elbo_vars, s, b, fsm_vec[b], sbs[s])
        end
    end
end



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
        linear_world_to_pix(ea.patches[s, b].wcs_jacobian,
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

                # There are no derivatives with respect to epsilon, so can
                # afely add to the value.
                E_G.v[1] += tile.epsilon_mat[pixel.h, pixel.w]

                # Note that with a lanczos_width > 1 negative values are
                # possible, and this will result in an error in \
                # add_elbo_log_term.

                # Add the terms to the elbo given the brightness.
                iota = tile.iota_vec[pixel.h]
                # println((b, h_fsm, w_fsm,
                #          E_G.v[1],
                #          fsms.fs0m_conv[h_fsm, w_fsm].v[1],
                #          fsms.fs1m_conv[h_fsm, w_fsm].v[1],
                #          tile.epsilon_mat[pixel.h, pixel.w]))
                add_elbo_log_term!(
                    elbo_vars, E_G, var_G, elbo_vars.elbo, this_pixel, iota)
                add_scaled_sfs!(elbo_vars.elbo, E_G, -iota,
                                elbo_vars.calculate_hessian &&
                                elbo_vars.calculate_derivs)

                # Subtract the log factorial term. This is not a function of the
                # parameters so the derivatives don't need to be updated. Note
                # that even though this does not affect the ELBO's maximum,
                # it affects the optimization convergence criterion, so I will
                # leave it in for now.
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


end
