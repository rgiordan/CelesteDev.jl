
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.get_expected_pixel_brightness!

using Celeste.Model.ParamSet
using SensitiveFloats.SensitiveFloat


"""
Produce a predicted image for a given tile and model parameters.
If include_epsilon is true, then the background is also rendered.
Otherwise, only pixels from the object are rendered.

Args:
    - tile: An ImageTile
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
    - include_epsilon: Whether the background noise should be included
"""
function get_expected_tile_brightness{NumType <: Number}(
                    tile::ImageTile,
                    ea::ElboArgs{NumType},
                    tile_sources::Vector{Int},
                    calculate_derivs::Bool,
                    calculate_hessian::Bool;
                    include_epsilon::Bool=false)

    elbo_vars =
        ElboIntermediateVariables(NumType, ea.S, length(ea.active_sources))
    elbo_vars.calculate_derivs = calculate_derivs
    elbo_vars.calculate_hessian = calculate_hessian
    star_mcs, gal_mcs =
      load_bvn_mixtures(ea, tile.b, calculate_derivs=calculate_derivs)
    sbs = load_source_brightnesses(ea, calculate_derivs=calculate_derivs)

    sf_type = typeof(elbo_vars.E_G)
    E_G_pixels = Array{sf_type}(size(tile.pixels))
    var_G_pixels = Array{sf_type}(size(tile.pixels))
    # Iterate over pixels that are not NaN.
    h_width, w_width = size(tile.pixels)
    for w in 1:w_width, h in 1:h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            get_expected_pixel_brightness!(
                elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
                ea, tile_sources, include_epsilon=include_epsilon)
            iota = tile.iota_vec[h]
            E_G_pixels[h, w] = deepcopy(elbo_vars.E_G)
            var_G_pixels[h, w] = deepcopy(elbo_vars.var_G)
        end
    end

    E_G_pixels, var_G_pixels
end


"""
Convolve and sum a matrix of sensitive floats with a matrix using FFT.

Args:
  - result_sf: Updated in place with the sum of the convolution
  - sf_matrix: A matrix of sensitive floats arranged spatially
  - float_matrix: Pre-allocated memory the same size as sf_matrix
  - conv_fft: The FFT of the signal you want to convolve, same size as sf_matrix
  - fft_plan: The plan for the FFT based on the size of sf_matrix.
"""
function convolve_and_add_sensitive_float!{ParamType <: ParamSet}(
    result_sf::SensitiveFloat{ParamType, Float64},
    sf_matrix::Matrix{SensitiveFloat{ParamType, Float64}},
    float_matrix::Matrix{Float64},
    conv_fft::Matrix{Complex{Float64}},
    fft_plan::Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,false,2})

    for ind in 1:length(float_matrix)
      float_matrix[ind] = 0
    end

    h_range = 1:size(sf_matrix, 1)
    w_range = 1:size(sf_matrix, 2)

    for h in h_range, w in w_range
      float_matrix[h, w] = sf_matrix[h, w].v[1]
    end
    result_sf.v[1] += sum(real(ifft(conv_fft .* (fft_plan * float_matrix))));

    for sa_d in 1:size(convolved_sf.d, 2), ind in 1:size(convolved_sf.d, 1)
      for h in h_range, w in w_range
        float_matrix[h, w] = sf_matrix[h, w].d[ind, sa_d]
      end
      result_sf.d[ind, sa_d] +=
        sum(real(ifft(conv_fft .* (fft_plan * float_matrix))));
    end

    for ind1 in 1:size(convolved_sf.h, 1), ind2 in 1:ind1
      for h in h_range, w in w_range
        float_matrix[h, w] = sf_matrix[h, w].h[ind2, ind1]
      end

      result_sf.h[ind1, ind2] += result_sf.h[ind2, ind1] =
        sum(real(ifft(conv_fft .* (fft_plan * float_matrix))));
    end
end
