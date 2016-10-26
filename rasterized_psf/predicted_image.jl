module PSFConvolution

using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.get_expected_pixel_brightness!
using Celeste.Model.populate_fsm_vecs!
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

typealias GMatrix Matrix{SensitiveFloat{CanonicalParams, Float64}}
typealias fs0mMatrix Matrix{SensitiveFloat{StarPosParams, Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{GalaxyPosParams, Float64}}

type FSMSensitiveFloatMatrices
    # The lower corner of the image (in terms of index values)
    h_lower::Int
    w_lower::Int

    fs0m_image::fs0mMatrix;
    fs1m_image::fs1mMatrix;
    fs0m_image_padded::fs0mMatrix;
    fs1m_image_padded::fs1mMatrix;
    fs0m_conv::fs0mMatrix;
    fs1m_conv::fs1mMatrix;
    fs0m_conv_padded::fs0mMatrix;
    fs1m_conv_padded::fs1mMatrix;

    E_G::GMatrix
    var_G::GMatrix

    psf_fft::Matrix{Complex{Float64}}

    # The amount of padding introduced by the convolution on one side of the
    # image (the total pixels added in each dimension are twice this)
    pad_pix_h::Int
    pad_pix_w::Int

    FSMSensitiveFloatMatrices() = begin
        new(1, 1,
            fs0mMatrix(), fs1mMatrix(),
            fs0mMatrix(), fs1mMatrix(),
            fs0mMatrix(), fs1mMatrix(),
            fs0mMatrix(), fs1mMatrix(),
            GMatrix(), GMatrix(),
            Matrix{Complex{Float64}}(),
            0, 0)
    end
end


function initialize_fsm_sf_matrices!(
    fsm_vec::Vector{FSMSensitiveFloatMatrices},
    ea::ElboArgs{Float64},
    psf_image_vec::Array{Matrix{Float64}})

    # Get the extreme active pixels in each band.
    h_lower = Int[typemax(Int) for b in ea.images ]
    w_lower = Int[typemax(Int) for b in ea.images ]
    h_upper = Int[0 for b in ea.images ]
    w_upper = Int[0 for b in ea.images ]

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
        fsm_vec[b].h_lower = h_lower[b]
        fsm_vec[b].w_lower = w_lower[b]

        psf_image = psf_image_vec[b]

        h_width = h_upper[b] - h_lower[b] + 1
        w_width = w_upper[b] - w_lower[b] + 1

        # An fsm value is only sensitive to one source's parameters.
        fsm_vec[b].fs0m_image = zero_sensitive_float_array(
            StarPosParams, Float64, 1, h_width, w_width);
        fsm_vec[b].fs1m_image = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, h_width, w_width);
        fsm_vec[b].fs0m_conv = zero_sensitive_float_array(
            StarPosParams, Float64, 1, h_width, w_width);
        fsm_vec[b].fs1m_conv = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, h_width, w_width);

        # Store the FFT of the psf image.
        (fft_size1, fft_size2) =
            (h_width + size(psf_image, 1) - 1, w_width + size(psf_image, 2) - 1)
        fsm_vec[b].psf_fft = zeros(Complex{Float64}, fft_size1, fft_size2);
        fsm_vec[b].psf_fft[1:size(psf_image, 1), 1:size(psf_image, 2)] = psf_image;
        fft!(fsm_vec[b].psf_fft);

        # The amount of padding introduced by the convolution.  Make sure
        # that the PSF has an odd dimension.
        @assert size(psf_image, 1) % 2 == 1
        @assert size(psf_image, 2) % 2 == 1
        fsm_vec[b].pad_pix_h = Integer((size(psf_image, 1) - 1) / 2)
        fsm_vec[b].pad_pix_w = Integer((size(psf_image, 2) - 1) / 2)

        fsm_vec[b].fs0m_image_padded = zero_sensitive_float_array(
            StarPosParams, Float64, 1, fft_size1, fft_size2);
        fsm_vec[b].fs1m_image_padded = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, fft_size1, fft_size2);

        fsm_vec[b].fs0m_conv_padded = zero_sensitive_float_array(
            StarPosParams, Float64, 1, fft_size1, fft_size2);
        fsm_vec[b].fs1m_conv_padded = zero_sensitive_float_array(
            GalaxyPosParams, Float64, 1, fft_size1, fft_size2);

        # Brightness images
        fsm_vec[b].E_G = zero_sensitive_float_array(
            CanonicalParams, Float64, sa_n, h_width, w_width);
        fsm_vec[b].var_G = zero_sensitive_float_array(
            CanonicalParams, Float64, sa_n, h_width, w_width);
    end
end


"""
Convolve a matrix of sensitive floats (represented by conv_fft) with a matrix of reals.

Args:
  - sf_matrix: A matrix of sensitive floats arranged spatially
  - float_matrix: Pre-allocated memory the same size as sf_matrix
  - conv_fft: The FFT of the signal you want to convolve, same size as sf_matrix
  - fft_plan: The plan for the FFT based on the size of sf_matrix.
"""
function convolve_sensitive_float_matrix!{ParamType <: ParamSet}(
    sf_matrix::Matrix{SensitiveFloat{ParamType, Float64}},
    conv_fft::Matrix{Complex{Float64}},
    sf_matrix_out::Matrix{SensitiveFloat{ParamType, Float64}})

    @assert size(sf_matrix) == size(conv_fft)

    # Pre-allocate memory.
    fft_matrix = zeros(Complex{Float64}, size(sf_matrix))
    n_active_sources = size(sf_matrix[1].d, 2)

    h_range = 1:size(sf_matrix, 1)
    w_range = 1:size(sf_matrix, 2)

    for h in h_range, w in w_range
      fft_matrix[h, w] = sf_matrix[h, w].v[1]
    end
    fft!(fft_matrix)
    fft_matrix .*= conv_fft
    ifft!(fft_matrix)
    for h in h_range, w in w_range
        sf_matrix_out[h, w].v[1] = real(fft_matrix[h, w]);
    end

    for sa_d in 1:n_active_sources, ind in 1:length(ParamType)
        for h in h_range, w in w_range
          fft_matrix[h, w] = sf_matrix[h, w].d[ind, sa_d]
        end
        fft!(fft_matrix)
        fft_matrix .*= conv_fft
        ifft!(fft_matrix)
        for h in h_range, w in w_range
            sf_matrix_out[h, w].d[ind, sa_d] = real(fft_matrix[h, w]);
        end
    end

    for ind1 in 1:size(sf_matrix[1].h, 1), ind2 in 1:ind1
        for h in h_range, w in w_range
          # TOOD: avoid this copy?
          fft_matrix[h, w] = sf_matrix[h, w].h[ind1, ind2]
        end
        fft!(fft_matrix)
        fft_matrix .*= conv_fft
        ifft!(fft_matrix)
        for h in h_range, w in w_range
            sf_matrix_out[h, w].h[ind1, ind2] = sf_matrix_out[h, w].h[ind2, ind1] =
                real(fft_matrix[h, w]);
        end
    end

    sf_matrix_out
end


"""
Convolve a populated set of SensitiveFloat matrices in fsms with the PSF
and store them in the matching fs*m_conv SensitiveFloat matrices.
"""
function convolve_fsm_images!(fsms::FSMSensitiveFloatMatrices)

    for h in 1:size(fsms.fs0m_image, 1), w in 1:size(fsms.fs0m_image, 2)
        fsms.fs0m_image_padded[h, w] = fsms.fs0m_image[h, w];
        fsms.fs1m_image_padded[h, w] = fsms.fs1m_image[h, w];
    end

    # conv_time = time()
    convolve_sensitive_float_matrix!(
        fsms.fs0m_image_padded, fsms.psf_fft, fsms.fs0m_conv_padded);
    convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft, fsms.fs1m_conv_padded);
    # conv_time = time() - conv_time
    # println("Convolution time: ", conv_time)

    for h in 1:size(fsms.fs0m_image, 1), w in 1:size(fsms.fs0m_image, 2)
        fsms.fs0m_conv[h, w] =
            fsms.fs0m_conv_padded[fsms.pad_pix_h + h, fsms.pad_pix_w + w];
        fsms.fs1m_conv[h, w] =
            fsms.fs1m_conv_padded[fsms.pad_pix_h + h, fsms.pad_pix_w + w];
    end

    # Set return type
    return true
end


function clear_fsms!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.fs0m_image clear!(sf) end
    for sf in fsms.fs1m_image clear!(sf) end
    for sf in fsms.fs0m_image_padded clear!(sf) end
    for sf in fsms.fs1m_image_padded clear!(sf) end
    for sf in fsms.fs0m_conv clear!(sf) end
    for sf in fsms.fs1m_conv clear!(sf) end
    for sf in fsms.fs0m_conv_padded clear!(sf) end
    for sf in fsms.fs1m_conv_padded clear!(sf) end
end


function clear_brightness!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.E_G clear!(sf) end
    for sf in fsms.var_G clear!(sf) end
end


"""
Populate the fsm shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in
fsms.fs0m_conv and fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_fsm_image!(
            ea::ElboArgs{Float64},
            elbo_vars::ElboIntermediateVariables{Float64},
            s::Int,
            b::Int,
            star_mcs::Array{BvnComponent{Float64}, 2},
            gal_mcs::Array{GalaxyCacheComponent{Float64}, 4},
            fsms::FSMSensitiveFloatMatrices)

    clear_fsms!(fsms)
    for pixel in ea.active_pixels
        if pixel.n == b
            # TODO: do this for all the sources.
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
            h_fsm = tile.h_range[pixel.h] - fsms.h_lower + 1
            w_fsm = tile.w_range[pixel.w] - fsms.w_lower + 1

            x = SVector{2, Float64}([tile.h_range[pixel.h], tile.w_range[pixel.w]])
            populate_fsm!(elbo_vars.bvn_derivs,
                          fsms.fs0m_image[h_fsm, w_fsm],
                          fsms.fs1m_image[h_fsm, w_fsm],
                          true, true,
                          s, x, true, Inf,
                          ea.patches[s, b].wcs_jacobian,
                          gal_mcs, star_mcs)
        end
    end
    convolve_fsm_images!(fsms);
end


"""
Updates fsms.E_G and fsms.var_G in place with the contributions from this
source in this band.
"""
function accumulate_source_band_brightness!(
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
            tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
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
    star_mcs_vec::Array{Array{BvnComponent{Float64}, 2}},
    gal_mcs_vec::Array{Array{GalaxyCacheComponent{Float64}, 4}},
    b::Int, include_epsilon::Bool)

    clear_brightness!(fsms)

    for s in 1:ea.S
        populate_fsm_image!(ea, elbo_vars, s, b, star_mcs_vec[s], gal_mcs_vec[s], fsms)
        accumulate_source_band_brightness!(ea, elbo_vars, s, b, fsms, sbs[s])
    end

    for pixel in ea.active_pixels
        if pixel.n == b
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            this_pixel = tile.pixels[pixel.h, pixel.w]

            # These are indices within the fs?m image.
            h_fsm = tile.h_range[pixel.h] - fsms.h_lower + 1
            w_fsm = tile.w_range[pixel.w] - fsms.w_lower + 1

            E_G = fsms.E_G[h_fsm, w_fsm]
            var_G = fsms.var_G[h_fsm, w_fsm]
            if include_epsilon
                # There are no derivatives with respect to epsilon, so can safely add
                # to the value.
                E_G.v[1] += tile.epsilon_mat[pixel.h, pixel.w]
            end

            # Add the terms to the elbo given the brightness.
            iota = tile.iota_vec[pixel.h]
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
