
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using DeterministicVI.ElboIntermediateVariables
using DeterministicVI.get_expected_pixel_brightness!


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
