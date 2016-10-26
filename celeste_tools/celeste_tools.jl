using Celeste: Model, DeterministicVI
using StaticArrays

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Distributions
using PyPlot


import Base.print
function print(ce::CatalogEntry)
  for field in fieldnames(ce)
    println(field, ": ", getfield(ce, field))
  end
end


type SubImage
    min_h::Int
    max_h::Int
    min_w::Int
    max_w::Int
    source_tiles::Array{Celeste.Model.ImageTile}
end


function get_source_pixel_range(sa::Int, b::Int, ea::Celeste.DeterministicVI.ElboArgs)
    source_tiles = ea.images[b].tiles[find([sa in tile for tile in ea.tile_source_map[b] ])];
    min_h = minimum([tile.h_range.start for tile in source_tiles])
    max_h = maximum([tile.h_range.stop for tile in source_tiles])
    min_w = minimum([tile.w_range.start for tile in source_tiles])
    max_w = maximum([tile.w_range.stop for tile in source_tiles])

    return SubImage(min_h, max_h, min_w, max_w, source_tiles)
end

function get_blank_image(si::SubImage)
    return Float64[NaN for h=1:(si.max_h - si.min_h + 1), w=1:(si.max_w - si.min_w + 1)];
end


function render_source(ea::ElboArgs, sa::Int, si::SubImage, include_epsilon::Bool)
    image = get_blank_image(si)
    for tile in si.source_tiles
      pix = DeterministicVI.tile_predicted_image(
        tile, ea, Int64[sa], include_epsilon=include_epsilon)
      image[tile.h_range - si.min_h + 1, tile.w_range - si.min_w + 1] = pix
    end
    return image
end


function show_source_image(ea::ElboArgs, sa::Int, si::SubImage)
    image = get_blank_image(si)
    for tile in si.source_tiles
        image[tile.h_range - si.min_h + 1, tile.w_range - si.min_w + 1] = tile.pixels
    end
    return image
end


function show_active_pixels(ea::ElboArgs, si::SubImage, b::Int)
    image = get_blank_image(si)
    for pixel in ea.active_pixels
        if pixel.n == b
            tile = ea.images[b].tiles[pixel.tile_ind]
            image[tile.h_range.start - si.min_h + pixel.h,
                  tile.w_range.start - si.min_w + pixel.w] = 1
        end
    end
    return image
end







# TODO: copy over
using SensitiveFloats.zero_sensitive_float_array
using SensitiveFloats.SensitiveFloat
using PSFConvolution.FSMSensitiveFloatMatrices
using PSFConvolution.initialize_fsm_sf_matrices!

using Celeste.DeterministicVI.ElboArgs
using Celeste.DeterministicVI.ElboIntermediateVariables
using DeterministicVI.BvnComponent
using DeterministicVI.GalaxyCacheComponent
using DeterministicVI.load_bvn_mixtures
using DeterministicVI.load_source_brightnesses
using Celeste.Model.populate_fsm!
using PSFConvolution.FSMSensitiveFloatMatrices
using PSFConvolution.accumulate_band_in_elbo!


function elbo_likelihood_with_fft!(
    ea::ElboArgs, elbo_vars::ElboIntermediateVariables,
    fsm_vec::Array{FSMSensitiveFloatMatrices})

    sbs = load_source_brightnesses(ea,
        calculate_derivs=elbo_vars.calculate_derivs,
        calculate_hessian=elbo_vars.calculate_hessian);

    star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, ea.N);
    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(ea, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian)
    end

    clear!(elbo_vars.elbo)
    for b in 1:ea.N
        accumulate_band_in_elbo!(
            ea, elbo_vars, fsm_vec[b], sbs, star_mcs_vec, gal_mcs_vec, b, true)
    end
end

using StaticArrays

using PSFConvolution.populate_fsm_image!
using PSFConvolution.accumulate_source_band_brightness!

function populate_fsm_vec!(
    ea::ElboArgs, elbo_vars::ElboIntermediateVariables,
    fsm_vec::Array{FSMSensitiveFloatMatrices})

    sbs = load_source_brightnesses(ea,
        calculate_derivs=elbo_vars.calculate_derivs,
        calculate_hessian=elbo_vars.calculate_hessian);

    star_mcs_vec = Array(Array{BvnComponent{Float64}, 2}, ea.N);
    gal_mcs_vec = Array(Array{GalaxyCacheComponent{Float64}, 4}, ea.N);
    for b=1:ea.N
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(ea, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian)
        for s in 1:ea.S
            populate_fsm_image!(
                ea, elbo_vars, s, b, star_mcs_vec[b], gal_mcs_vec[b], fsm_vec[b])
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
