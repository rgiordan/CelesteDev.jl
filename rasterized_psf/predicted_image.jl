
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
function convolve_sensitive_float_matrix{ParamType <: ParamSet}(
    sf_matrix::Matrix{SensitiveFloat{ParamType, Float64}},
    conv_fft::Matrix{Complex{Float64}})

    @assert size(sf_matrix) == size(conv_fft)

    # Pre-allocate memory.
    fft_matrix = zeros(Complex{Float64}, size(sf_matrix))
    n_active_sources = size(sf_matrix[1].d, 2)
    sf_matrix_out =
        SensitiveFloat{ParamType, Float64}[
            zero_sensitive_float(ParamType, Float64, n_active_sources) for sf in sf_matrix]

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
