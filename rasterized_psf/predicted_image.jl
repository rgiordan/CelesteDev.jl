
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

function fsm_from_active_pixels!{NumType <: Number}(
                elbo_vars::ElboIntermediateVariables{NumType},
                b::Int,
                ea::ElboArgs{NumType},
                active_pixels::Array{ActivePixel})

    # sbs = load_source_brightnesses(ea,
    #     calculate_derivs=elbo_vars.calculate_derivs,
    #     calculate_hessian=elbo_vars.calculate_hessian)

    star_mcs_vec = Array(Array{BvnComponent{NumType}, 2}, ea.N)
    gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType}, 4}, ea.N)

    for b=1:ea.N
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(ea, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian)
    end

    # iterate over the pixels
    for pixel in active_pixels
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # Get the brightness.
        get_expected_pixel_brightness!(
            elbo_vars, pixel.h, pixel.w, sbs,
            star_mcs_vec[pixel.n], gal_mcs_vec[pixel.n], tile,
            ea, tile_sources, include_epsilon=true)

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
    end
end




"""
Do it
"""
function get_source_tile_fsm_matrix{NumType <: Number}(
                    tile::ImageTile,
                    ea::ElboArgs{NumType},
                    s::Integer,
                    elbo_vars::ElboIntermediateVariables{NumType},
                    star_mcs::Array{BvnComponent{NumType}, 2},
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    tile_sources::Vector{Int},
                    calculate_derivs::Bool,
                    calculate_hessian::Bool;
                    include_epsilon::Bool=false)

    fs0m_tile = Array{SensitiveFloat{StarPosParams, NumType}}(size(tile.pixels))
    fs1m_tile = Array{SensitiveFloat{GalaxyPosParams, NumType}}(size(tile.pixels))

    h_width, w_width = size(tile.pixels)
    for w in 1:w_width, h in 1:h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
              # TODO: Have this store all sources.
              populate_fsm_vecs!(
                  elbo_vars, ea, tile_sources, tile, h, w, gal_mcs, star_mcs)

              fs0m_tile[h, w] = deepcopy(elbo_vars.fs0m_vec[s])
              fs1m_tile[h, w] = deepcopy(elbo_vars.fs1m_vec[s])
        end
    end

    fs0m_tile, fs1m_tile
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



function convolve_fsm_images!{NumType <: Number}(
    fs0m_image::Array{SensitiveFloat{StarPosParams, NumType}, 2},
    fs1m_image::Array{SensitiveFloat{GalaxyPosParams, NumType}, 2},
    fs0m_image_padded::Array{SensitiveFloat{StarPosParams, NumType}, 2},
    fs1m_image_padded::Array{SensitiveFloat{GalaxyPosParams, NumType}, 2},
    fs0m_conv::Array{SensitiveFloat{StarPosParams, NumType}, 2},
    fs1m_conv::Array{SensitiveFloat{GalaxyPosParams, NumType}, 2},
    fs0m_conv_padded::Array{SensitiveFloat{StarPosParams, NumType}, 2},
    fs1m_conv_padded::Array{SensitiveFloat{GalaxyPosParams, NumType}, 2},
    psf_fft::Matrix{Complex{Float64}})

    for h in 1:size(fs0m_image, 1), w in 1:size(fs0m_image, 2)
        fs0m_image_padded[h, w] = fs0m_image[h, w];
        fs1m_image_padded[h, w] = fs1m_image[h, w];
    end

    conv_time = time()
    convolve_sensitive_float_matrix!(fs0m_image_padded, psf_fft, fs0m_conv_padded);
    convolve_sensitive_float_matrix!(fs1m_image_padded, psf_fft, fs1m_conv_padded);
    conv_time = time() - conv_time
    println("Convolution time: ", conv_time)

    pad_pix_h = Integer((size(fs0m_image_padded, 1) - size(fs0m_image, 1)) / 2)
    pad_pix_w = Integer((size(fs0m_image_padded, 2) - size(fs0m_image, 2)) / 2)

    fs0m_conv = fs0m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h),
                                 (pad_pix_w + 1):(end - pad_pix_w)];
    fs1m_conv = fs1m_conv_padded[(pad_pix_h + 1):(end - pad_pix_h),
                                 (pad_pix_w + 1):(end - pad_pix_w)];

    return fs0m_conv, fs1m_conv
end

end
