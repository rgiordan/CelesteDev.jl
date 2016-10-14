
module PSFConvolution

using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.get_expected_pixel_brightness!
using DeterministicVI.populate_fsm_vecs!
using Celeste.Model.ParamSet
using SensitiveFloats.SensitiveFloat
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent

using SensitiveFloats.zero_sensitive_float
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using SensitiveFloats.clear!

using DeterministicVI.accumulate_source_brightness!
using DeterministicVI.add_sources_sf!
using DeterministicVI.add_elbo_log_term!
using DeterministicVI.add_scaled_sfs!
using DeterministicVI.ElboArgs
using DeterministicVI.ActivePixel
using DeterministicVI.ImageTile
using DeterministicVI.SourceBrightness
using DeterministicVI.StarPosParams
using DeterministicVI.GalaxyPosParams


typealias fs0mMatrix Matrix{SensitiveFloat{StarPosParams, Float64}}
typealias fs1mMatrix Matrix{SensitiveFloat{GalaxyPosParams, Float64}}

type FSMSensitiveFloatMatrices
    fs0m_image::fs0mMatrix;
    fs1m_image::fs1mMatrix;
    fs0m_image_padded::fs0mMatrix;
    fs1m_image_padded::fs1mMatrix;
    fs0m_conv::fs0mMatrix;
    fs1m_conv::fs1mMatrix;
    fs0m_conv_padded::fs0mMatrix;
    fs1m_conv_padded::fs1mMatrix;
    psf_fft::Matrix{Complex{Float64}};

    FSMSensitiveFloatMatrices() = begin
        new(fs0mMatrix(), fs1mMatrix(),
            fs0mMatrix(), fs1mMatrix(),
            fs0mMatrix(), fs1mMatrix(),
            fs0mMatrix(), fs1mMatrix(),
            Matrix{Complex{Float64}}())
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
        psf_image = psf_image_vec[b]

        h_width = h_upper[b] - h_lower[b] + 1
        w_width = w_upper[b] - w_lower[b] + 1

        fsm_vec[b].fs0m_image = zero_sensitive_float_array(
            StarPosParams, Float64, sa_n, h_width, w_width);
        fsm_vec[b].fs1m_image = zero_sensitive_float_array(
            GalaxyPosParams, Float64, sa_n, h_width, w_width);
        fsm_vec[b].fs0m_conv = zero_sensitive_float_array(
            StarPosParams, Float64, sa_n, h_width, w_width);
        fsm_vec[b].fs1m_conv = zero_sensitive_float_array(
            GalaxyPosParams, Float64, sa_n, h_width, w_width);

        (fft_size1, fft_size2) =
            (h_width + size(psf_image, 1) - 1, w_width + size(psf_image, 2) - 1)
        fsm_vec[b].psf_fft = zeros(Complex{Float64}, fft_size1, fft_size2);
        fsm_vec[b].psf_fft[1:size(psf_image, 1), 1:size(psf_image, 2)] = psf_image;
        fft!(fsm_vec[b].psf_fft);

        fsm_vec[b].fs0m_image_padded =
            zero_sensitive_float_array(StarPosParams, Float64, sa_n,
            fft_size1, fft_size2);
        fsm_vec[b].fs1m_image_padded =
            zero_sensitive_float_array(GalaxyPosParams, Float64, sa_n,
            fft_size1, fft_size2);

        fsm_vec[b].fs0m_conv_padded =
            zero_sensitive_float_array(StarPosParams, Float64, sa_n,
            fft_size1, fft_size2);
        fsm_vec[b].fs1m_conv_padded =
            zero_sensitive_float_array(GalaxyPosParams, Float64, sa_n,
            fft_size1, fft_size2);
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


function convolve_sensitive_float_matrix{ParamType <: ParamSet}(
    sf_matrix::Matrix{SensitiveFloat{ParamType, Float64}},
    conv_fft::Matrix{Complex{Float64}})

    sf_matrix_out =
        SensitiveFloat{ParamType, Float64}[
            zero_sensitive_float(ParamType, Float64, n_active_sources) for sf in sf_matrix]

    convolve_sensitive_float_matrix!(sf_matrix, conv_fft, sf_matrix_out)

    sf_matrix_out
end


function set_point_psf!(ea::ElboArgs, point_psf_width::Float64)
    point_psf = Model.PsfComponent(1.0, Float64[0, 0],
        Float64[ point_psf_width 0.0; 0.0 point_psf_width ])
    for s in 1:size(ea.patches)[1], b in 1:size(ea.patches)[2]
      ea.patches[s, b] = SkyPatch(ea.patches[s, b], Model.PsfComponent[ point_psf ]);
    end
    ea.psf_K = 1
    return true
end



function get_expected_brightness_from_image!{NumType <: Number}(
    ea::ElboArgs{NumType},
    elbo_vars::ElboIntermediateVariables{NumType},
    active_pixels::Array{ActivePixel},
    b::Int, # TODO iterate over these
    s::Int,
    sbs::Vector{SourceBrightness{NumType}},
    star_mcs::Array{BvnComponent{NumType}, 2},
    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
    fs0m_conv::Array{SensitiveFloat{StarPosParams, NumType}, 2},
    fs1m_conv::Array{SensitiveFloat{GalaxyPosParams, NumType}, 2},
    h_lower::Array{Int}, w_lower::Array{Int},
    include_epsilon::Bool=true)

    for pixel in active_pixels
        if pixel.n == b
            tile = ea.images[pixel.n].tiles[pixel.tile_ind]
            tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
            this_pixel = tile.pixels[pixel.h, pixel.w]

            # Get the brightness.

            # get_expected_pixel_brightness!(
            #     elbo_vars, pixel.h, pixel.w, sbs,
            #     star_mcs_vec[pixel.n], gal_mcs_vec[pixel.n], tile,
            #     ea, tile_sources, include_epsilon=true)

            # This combines the sources into a single brightness value for the pixel.
            # combine_pixel_sources!(elbo_vars, ea, tile_sources, tile, sbs)

            clear!(elbo_vars.E_G,
                elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)
            clear!(elbo_vars.var_G,
                elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)

            # For now s is fixed, and only doing one band.
            # for s in tile_sources
            active_source = s in ea.active_sources
            calculate_hessian =
                elbo_vars.calculate_hessian && elbo_vars.calculate_derivs &&
                active_source

            # The key is this: ea.fs?m_vec[s] must contain the appropriate fsm
            # sensitive floats.

            # These are indices within the fs?m image.
            h = tile.h_range[pixel.h] - h_lower[b] + 1
            w = tile.w_range[pixel.w] - w_lower[b] + 1

            elbo_vars.fs0m_vec[s] = fs0m_conv[h, w];
            elbo_vars.fs1m_vec[s] = fs1m_conv[h, w];
            accumulate_source_brightness!(elbo_vars, ea, sbs, s, tile.b);

            if active_source
                sa = findfirst(ea.active_sources, s)
                add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, sa, calculate_hessian)
                add_sources_sf!(elbo_vars.var_G, elbo_vars.var_G_s, sa, calculate_hessian)
            else
                # If the sources is inactives, simply accumulate the values.
                elbo_vars.E_G.v[1] += elbo_vars.E_G_s.v[1]
                elbo_vars.var_G.v[1] += elbo_vars.var_G_s.v[1]
            end

            # Add the terms to the elbo given the brightness.
            iota = tile.iota_vec[pixel.h]
            add_elbo_log_term!(elbo_vars, this_pixel, iota)
            add_scaled_sfs!(elbo_vars.elbo,
                            elbo_vars.E_G, -iota,
                            elbo_vars.calculate_hessian &&
                            elbo_vars.calculate_derivs)

            # Subtract the log factorial term. This is not a function of the
            # parameters so the derivatives don't need to be updated. Note that
            # even though this does not affect the ELBO's maximum, it affects
            # the optimization convergence criterion, so I will leave it in for now.
            elbo_vars.elbo.v[1] -= lfact(this_pixel)

            if include_epsilon
                # There are no derivatives with respect to epsilon, so can safely add
                # to the value.
                elbo_vars.E_G.v[1] += tile.epsilon_mat[h, w]
            end
        end
    end
end



function convolve_fsm_images!{NumType <: Number}(fsms::FSMSensitiveFloatMatrices)

    for h in 1:size(fsms.fs0m_image, 1), w in 1:size(fsms.fs0m_image, 2)
        fsms.fs0m_image_padded[h, w] = fsms.fs0m_image[h, w];
        fsms.fs1m_image_padded[h, w] = fsms.fs1m_image[h, w];
    end

    conv_time = time()
    convolve_sensitive_float_matrix!(
        fsms.fs0m_image_padded, fsms.psf_fft, fsms.fs0m_conv_padded);
    convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft, fsms.fs1m_conv_padded);
    conv_time = time() - conv_time
    println("Convolution time: ", conv_time)

    pad_pix_h = Integer((size(fsms.fs0m_image_padded, 1) - size(fsms.fs0m_image, 1)) / 2)
    pad_pix_w = Integer((size(fsms.fs0m_image_padded, 2) - size(fsms.fs0m_image, 2)) / 2)

    fsms.fs0m_conv = fsms.fs0m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h),
                                           (pad_pix_w + 1):(end - pad_pix_w)];
    fsms.fs1m_conv = fsms.fs1m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h),
                                           (pad_pix_w + 1):(end - pad_pix_w)];

    return true
end




end
